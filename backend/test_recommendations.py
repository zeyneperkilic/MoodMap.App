import requests

API = "http://localhost:8000"

def test_recommendations():
    failures = []
    for cluster in range(4):
        for intensity in (1, 5, 10):
            resp = requests.get(f"{API}/recommend/{cluster}", params={"intensity": intensity})
            if resp.status_code != 200:
                failures.append(f"HTTP {resp.status_code} for cluster={cluster},intensity={intensity}")
                continue
            data = resp.json()
            n = len(data.get("songs", []))
            # expect at least 1, ideally 10
            if n == 0:
                failures.append(f"No songs returned for cluster={cluster},intensity={intensity}")
            else:
                print(f"✅ Cluster {cluster}, Intensity {intensity} → {n} songs")

    if failures:
        print("\n❌ Failures:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    else:
        print("\n🎉 All recommendation tests passed!")

if __name__=="__main__":
    test_recommendations()