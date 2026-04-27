# Fix Microphone Blocked in Browser

Browsers block microphone access on **HTTP** (non-secure) pages. Only **HTTPS** or **localhost** are allowed.

## Quick fix: use localhost

If you access the UI at `http://192.168.1.163:3000` from the same PC, the mic is blocked.

**Solution:** Open `http://localhost:3000` instead. Add a hosts entry so `localhost` points to the Pi, or run the UI locally:

```powershell
cd C:\Users\DiegoTorres\Desktop\NeuroLinux\phase4-distributed\neurohub-ui
npm run dev
```

Then open `http://localhost:3000/voice` — mic will work.

## Alternative: Pi Mic mode (no browser mic)

- **Dance:** Use **Mode B (Pi Mic)** on the Cosmos page — Pi's C270 mic captures music, no browser mic needed.
- **Voice:** Speech recognition requires browser mic; use Pi Mic mode for dance when on HTTP.

## Chrome/Edge site settings

If you did deny the mic, fix it:

1. Click the **padlock** (or info icon) in the address bar
2. **Site settings** → **Microphone** → **Allow**
3. Reload the page

## Hosts entry (optional)

To reach Pi via a localhost-like name:

```
# C:\Windows\System32\drivers\etc\hosts
192.168.1.163 neurolinux.local
```

Then `http://neurolinux.local:3000` still won't be secure; browsers treat it as HTTP. Use `localhost` with the UI running on your PC for voice/mic features.
