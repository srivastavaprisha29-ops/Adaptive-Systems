# Next research phase: width and precision action profiling

**Status: the frozen compact multi-load latency validation is complete. Quality noninferiority remains inconclusive per load; fixed-baseline promotion awaits a larger independent quality panel.** The frozen load sweep found no supported winning range for queue-aware confidence relaxation. Carry the calibrated fixed margin policy forward and return to the project's original representation question.

The [matched live PCA experiment](live-pca-comparison.md) selected 256D, halved vector bytes, but failed held-out quality preservation and live latency gates. Both paths still execute the same ResNet-18 graph. Target actual encoder computation next, with matched training and static quality/cost checks before adaptation. Native encoder precision changes remain untested in this phase.

The [four-configuration study](width-precision-grid.md) found no supported load-dependent crossover. INT8 is timed feature storage with FP32 reconstruction. Its conditional compact-encoder control passed the matched pilot quality/latency gates, while changing architecture as well as channel widths. No width/precision controller or production replacement is adopted. Native encoder arithmetic precision remains untested.

The [structurally narrowed ResNet experiment](channel-narrow-encoder.md) now tests final-stage internal channels directly. It confirms smaller convolution tensors but fails calibration and held-out quality/latency gates under weight-energy pruning plus BN/ridge adaptation. This motivates accuracy-recovery training before claiming a same-family narrow-encoder win; it does not overturn the separate compact-architecture pilot.

## Research question

Do quality-feasible width/precision configurations offer a measured cost or resource advantage after projection, conversion and dispatch costs, and are there distinct regimes where adaptation beats the strongest fixed configuration?

## Immediate work

1. Preserve `configs/carry-forward-fixed.json` as the inference reference. Freeze model family, checkpoints, task quality tolerance, dataset splits and deployment boundary before selecting representation actions.
2. Inventory compatible trained widths from the existing representation implementations. Compare width-only and precision-only controls as well as their cross-product. Every action needs a head trained for its representation; dropping arbitrary coordinates is insufficient.
3. Separate storage precision from compute precision. Label INT8 storage followed by FP32 reconstruction as storage compression. Include a low-precision arithmetic action only after verifying that its actual operators/backend support it on the named machine.
4. Profile the complete inference boundary and component costs with paired timing. Record encoder, projection, quantize/dequantize, head, dispatch and any measured transfer costs. Address the wall/thread-CPU outliers and timing variability seen in the controller experiments before interpreting small latency differences.
5. Measure raw feature bytes separately from allocated/peak system memory. Do not infer total memory or energy savings from tensor width. Any energy claim requires its own instrumented measurement.
6. Use calibration to select a strongest feasible fixed representation before evaluating adaptation. Freeze selection and test on reserved new traces or panels, with paired uncertainty and explicit quality checks.

## Go/no-go gate

Do not build a new adaptive policy until action profiles show at least two compatible, quality-feasible choices with credible differences in cost or measured resource-budget feasibility. A latency win retains the existing 5% lower end-to-end p95 and at most one-point accuracy-loss criterion against the relevant strongest fixed comparator. A separate storage/memory result must state its own measured objective and must not be presented as a latency win.

If the shared encoder dominates runtime, a smaller pooled representation may improve payload size without materially improving complete inference. Report that boundary honestly. The next implementation decision should follow the measured action surface, rather than more queue-threshold tuning on already evaluated traces.

## Fresh compact architecture validation

The [frozen multi-load study](compact-load-sweep.md) completes thirty matched sessions across five nominal load points plus a separately reported slow-load check. Compact clears every original latency comparison, but the original per-load quality bounds remain inconclusive. The supplemental low-load check exposed an unresolved encoder spike. No fixed-baseline promotion or controller is adopted; independent timing stability and larger quality validation are next.
