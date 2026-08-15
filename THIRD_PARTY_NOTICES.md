# Third-Party Notices

## Knowhere

The official repository at `https://github.com/Ontos-AI/knowhere.git` was refreshed
for the T9-4.4.1 audit and pinned at commit
`2e4eb5846249d273b11902ee00f26db949e45b38`. The upstream project carries the
Apache License 2.0 and its NOTICE identifies Knowhere API, Copyright 2026 Ontos-AI.
The previous reference audit used commit
`8d2bb0d4edda2074f1fad084040b183c8a7522ab`.

No Knowhere source is vendored as of T9-4.4.1. The exact source lock, license hashes,
and audited Git blobs are recorded in
`docs/evidence/t9-4-4-1/knowhere-source-lock.json`. If a later task copies or modifies
upstream source, the copied paths, Apache-2.0 license text, NOTICE requirements, and
local changes must be recorded here before that code is committed.

## T9-4.4.3 document parsing libraries

The project directly depends on the following Python packages for local document and
image parsing. Versions are fixed by `requirements.lock`; their source code is not
vendored into this repository.

| Package | Version validated in T9-4.4.3 | License |
| --- | --- | --- |
| beautifulsoup4 | 4.15.0 | MIT |
| lxml | 6.1.1 | BSD-3-Clause |
| markdownify | 1.2.3 | MIT |
| openpyxl | 3.1.5 | MIT |
| Pillow | 12.3.0 | MIT-CMU |
| python-docx | 1.2.0 | MIT |
| python-pptx | 1.0.2 | MIT |

Transitive runtime packages and exact artifact hashes are recorded in
`requirements.lock`. The adapters are project-owned implementations that use these
public APIs. Knowhere was consulted at the pinned commit for block-order and asset
accumulation ideas, but no Knowhere source file was copied in T9-4.4.3.
