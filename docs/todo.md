# Texture Studio todo

## Low priority: compact the full-resolution comparison layout

Requested October 12, 2026. References: `/opt/ipde/remove_part.png` and
`/opt/ipde/remove_part2.png`.

- Remove the oversized dark-blue header area and unused vertical space so the
  comparison images begin higher in the window.
- Arrange zoom, display controls, comparison selection, source-agreement
  metrics, and export actions in a compact header. Let controls wrap when the
  window is narrow instead of reserving a large blank region.
- Wrap map panes into additional rows when many checkpoints are visible, keeping
  each image useful for inspection instead of squeezing every map into one row.
- Preserve the source-reference pane, synchronized pan/zoom, shared display
  settings, readable metrics, and access to each map's actions. Export must keep
  original dimensions, numeric precision, and bytes.

Implementation entry point: `src/TextureStudio/Views/ReviewWorkbenchView.swift`.
Check a source plus one prediction, a ten-map comparison, and narrow/wide
windows. This item is deferred; no runtime layout change is included here.
