# Bundled documentation

The Help menu serves `help/` and `manual/` through the application's Qt Network loopback server. The guide works offline with no CDN, analytics, remote scripts or external server installation. Links to GitHub and reference documentation open external sites only when selected. Keep the app running while using its local help address.

Update `manual/chapters/*.md` as features change, then run `python3 scripts/build-docs.py`. The script generates `manual/manual.md` and the printable `manual/index.html` using only the Python standard library. These files are checked in so app builds require no Markdown tooling. Run `python3 scripts/build-docs.py --check` to detect stale generated output.

Update the approachable guide in `help/index.html`, its interaction logic in `help/help.js`, and styles in `help/style.css`. Each topic has an app-role anchor used by Help menu routing (`studio`, `extractor`, `photo`, `raw`, `datasets`, `trainer`). Keep sentences short, define specialist terms before use, and explain each choice through a concrete outcome. The browser's Print command prints the manual; its HTML and editable Markdown formats are both bundled.

When a feature changes, update its manual chapter, guide topic, glossary and interactive calculation where applicable. Verify link targets, calculator arithmetic, keyboard access and a narrow-screen layout. Numerical examples must state the native camera grid, label units, whether values are estimated/measured, and whether resampling is involved. Keep release compatibility guidance in sync with the release policy.
