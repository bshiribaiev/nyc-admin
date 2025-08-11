import json, subprocess, sys

def run_query(q):
    p = subprocess.run([sys.executable, "live_rag.py"],
                       input=(q+"\n").encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "ignore")
    # naive scrape of final answer line(s)
    ans = out.split("=== ANSWER ===")[-1].strip()
    return ans, out

def main():
    tests = json.load(open("./test.json"))
    ok = 0
    for t in tests:
        ans, raw = run_query(t["query"])
        if t.get("expect_no_answer"):
            passed = ans.strip() == "NO_ANSWER"
        else:
            passed = any(sec in ans for sec in t["must_contain_sections"])
        print(("✓" if passed else "✗"), t["query"])
        if not passed:
            print(ans)
        ok += int(passed)
    print(f"\nPassed {ok}/{len(tests)}")

if __name__ == "__main__":
    main()
