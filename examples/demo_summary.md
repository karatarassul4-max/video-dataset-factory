# Dataset Summary

This is a small synthetic fixture showing the reporting format expected from a real run. Replace it with numbers from `vdf process-folder`, `vdf dedupe-manifest`, and `vdf summarize-manifest` before using the project in an application.

| Metric | Value |
| --- | ---: |
| Total clips | 6 |
| Accepted clips | 3 |
| Rejected clips | 3 |
| Acceptance rate | 50.0% |
| Near-duplicate clips | 1 |
| Average aesthetic score | 6.95 |
| Average motion score | 2.36 |

## Reject Reasons

| Reason | Count |
| --- | ---: |
| `motion_too_low` | 1 |
| `near_duplicate` | 1 |
| `text_or_watermark_likely` | 1 |

## Example Interpretation

- The yield is balanced enough for dashboard smoke testing: accepted, rejected, and duplicate cases are all represented.
- `near_duplicate` demonstrates manifest-level pHash filtering after raw clip processing.
- `motion_too_low` and `text_or_watermark_likely` demonstrate auditable rejection reasons that are useful during dataset review.
