# scan.py
with open("tts_engine/s2.exe", "rb") as f:
    # Read the raw binary and decode it to text
    data = f.read().decode("ascii", errors="ignore")

print("\n--- Hardcoded API Endpoints in s2.exe ---")
# C++ strings are null-terminated (\x00). We split by them to find exact strings.
for word in set(data.split('\x00')):
    # Filter for standard URL paths
    if word.startswith('/') and len(word) < 25 and '\n' not in word and ' ' not in word:
        if not word.startswith('/usr') and not word.startswith('/lib'):
            print(f"Found Path: {word}")