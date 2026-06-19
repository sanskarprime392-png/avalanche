"""
Download C-SNOW High-Mountain-Asia Sentinel-1 snow-depth NetCDFs over SFTP into Drive.

Remote layout (discovered): data/High_Mountain_Asia/snd_YYYYMMDD.nc  — 1278 files, ~5 GB,
one ~1 km HMA-wide grid per Sentinel-1 date (2016-2020). Whole-HMA grids; clip to the AOI
later in processing. Skips files already present, so it is resumable.

Credentials are NOT stored in the repo. In Colab add them via the 🔑 Secrets panel (toggle
"Notebook access"):  CSNOW_USER = sentinel1snow ,  CSNOW_PASS = <password from the access email>.
Or export env vars CSNOW_USER / CSNOW_PASS.

Usage (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install paramiko
    %run data_access/fetch_csnow.py
"""
import os
import stat
import time
import paramiko
import config

HOST, PORT = "hydras.ugent.be", 2225
REMOTE_DIR = "data/High_Mountain_Asia"
OUT_DIR = os.path.join(config.PATHS["predictors"], "snow", "csnow_hma")

# Optional date window as YYYYMMDD ints (None = all). Set both to restrict to winter, etc.
DATE_MIN, DATE_MAX = None, None


def _creds():
    user, pw = os.environ.get("CSNOW_USER"), os.environ.get("CSNOW_PASS")
    if not (user and pw):
        try:
            from google.colab import userdata
            user = user or userdata.get("CSNOW_USER")
            pw = pw or userdata.get("CSNOW_PASS")
        except Exception:
            pass
    if not (user and pw):
        raise RuntimeError("Set CSNOW_USER / CSNOW_PASS in Colab Secrets (🔑) or env vars.")
    return user, pw


def _connect():
    user, pw = _creds()
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=user, password=pw)
    return t, paramiko.SFTPClient.from_transport(t)


def _in_window(fname):
    if DATE_MIN is None and DATE_MAX is None:
        return True
    try:
        d = int(fname.replace("snd_", "").replace(".nc", ""))
    except ValueError:
        return True
    return not ((DATE_MIN and d < DATE_MIN) or (DATE_MAX and d > DATE_MAX))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t, sftp = _connect()
    try:
        files = sorted(
            e.filename for e in sftp.listdir_attr(REMOTE_DIR)
            if not stat.S_ISDIR(e.st_mode) and e.filename.endswith(".nc") and _in_window(e.filename)
        )
        print(f"{len(files)} files to consider in {REMOTE_DIR} -> {OUT_DIR}")
        done = skipped = failed = 0
        for i, f in enumerate(files, 1):
            local = os.path.join(OUT_DIR, f)
            if os.path.exists(local) and os.path.getsize(local) > 0:
                skipped += 1
                continue
            remote = f"{REMOTE_DIR}/{f}"
            for attempt in (1, 2, 3):
                try:
                    sftp.get(remote, local + ".part")
                    os.replace(local + ".part", local)
                    done += 1
                    break
                except Exception as ex:
                    if attempt == 3:
                        failed += 1
                        print(f"  FAILED {f}: {ex}")
                    else:
                        time.sleep(2 * attempt)
                        try:                       # reconnect on a dropped transport
                            sftp.close(); t.close()
                        except Exception:
                            pass
                        t, sftp = _connect()
            if i % 50 == 0:
                try:                       # proactive reconnect — this server throttles a long-lived
                    sftp.close(); t.close()  # channel after a few hundred gets ("Garbage packet received")
                except Exception:
                    pass
                t, sftp = _connect()
                print(f"  {i}/{len(files)}  (new {done}, skipped {skipped}, failed {failed})")
        print(f"\nDone. downloaded {done}, skipped {skipped}, failed {failed}.  -> {OUT_DIR}")
    finally:
        try:
            sftp.close(); t.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
