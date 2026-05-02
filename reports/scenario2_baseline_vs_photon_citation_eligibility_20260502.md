# Scenario 2 Baseline vs PHOTON Score Report

- Matched turns: 51
- Max score per turn: 10
- Baseline avg score: 8.98
- PHOTON avg score: 9.941
- Baseline avg latency: 11804.1 ms
- PHOTON avg latency: 9749.8 ms
- Winner counts: `{"photon_latency": 19, "photon": 15, "tie": 15, "baseline_latency": 1, "baseline": 1}`

| Turn | Winner | Baseline | PHOTON | Delta | Notes |
|---|---|---:|---:|---:|---|
| chain_A2_A3_A5:A2:T1 | photon_latency | 10 | 10 | 0 |  |
| chain_A2_A3_A5:A2:T2 | photon | 7 | 10 | 3 | answer_missing=1;evidence_missing=1 |
| chain_A2_A3_A5:A2:T3 | photon_latency | 10 | 10 | 0 |  |
| chain_A2_A3_A5:A3:T1 | photon_latency | 10 | 10 | 0 |  |
| chain_A2_A3_A5:A3:T2 | photon | 9 | 10 | 1 | evidence_missing=1 |
| chain_A2_A3_A5:A3:T3 | photon_latency | 10 | 10 | 0 |  |
| chain_A2_A3_A5:A5:T1 | photon_latency | 10 | 10 | 0 |  |
| chain_A2_A3_A5:A5:T2 | photon_latency | 10 | 10 | 0 |  |
| chain_A2_A3_A5:A5:T3 | photon | 4 | 10 | 6 | forbidden_answer;evidence_missing=1;forbidden_cited_paths=2;too_many_citations=7>6 |
| single_A1:A1:T1 | tie | 10 | 10 | 0 |  |
| single_A1:A1:T2 | photon | 9 | 10 | 1 | evidence_missing=1 |
| single_A1:A1:T3 | photon | 8 | 10 | 2 | evidence_missing=1;forbidden_cited_paths=1 |
| single_A2:A2:T1 | photon_latency | 10 | 10 | 0 |  |
| single_A2:A2:T2 | photon | 7 | 10 | 3 | answer_missing=1;evidence_missing=1 |
| single_A2:A2:T3 | photon_latency | 10 | 10 | 0 |  |
| single_A3:A3:T1 | tie | 10 | 10 | 0 |  |
| single_A3:A3:T2 | photon | 9 | 10 | 1 | evidence_missing=1 |
| single_A3:A3:T3 | baseline_latency | 10 | 10 | 0 |  |
| single_A4:A4:T1 | tie | 10 | 10 | 0 |  |
| single_A4:A4:T2 | photon_latency | 10 | 10 | 0 |  |
| single_A4:A4:T3 | photon_latency | 10 | 10 | 0 |  |
| single_A5:A5:T1 | tie | 10 | 10 | 0 |  |
| single_A5:A5:T2 | photon_latency | 10 | 10 | 0 |  |
| single_A5:A5:T3 | photon | 4 | 10 | 6 | forbidden_answer;evidence_missing=1;forbidden_cited_paths=2;too_many_citations=8>6 |
| single_B1:B1:T1 | tie | 10 | 10 | 0 |  |
| single_B1:B1:T2 | photon_latency | 10 | 10 | 0 |  |
| single_B1:B1:T3 | photon_latency | 10 | 10 | 0 |  |
| single_B2:B2:T1 | tie | 10 | 10 | 0 |  |
| single_B2:B2:T2 | photon | 6 | 10 | 4 | evidence_missing=1;forbidden_cited_paths=2;too_many_citations=8>5 |
| single_B2:B2:T3 | tie | 10 | 10 | 0 |  |
| single_B3:B3:T1 | tie | 10 | 10 | 0 |  |
| single_B3:B3:T2 | photon | 8 | 10 | 2 | evidence_missing=1 |
| single_B3:B3:T3 | photon_latency | 10 | 10 | 0 |  |
| single_B4:B4:T1 | tie | 10 | 10 | 0 |  |
| single_B4:B4:T2 | photon_latency | 10 | 10 | 0 |  |
| single_B4:B4:T3 | photon_latency | 10 | 10 | 0 |  |
| single_B5:B5:T1 | tie | 10 | 10 | 0 |  |
| single_B5:B5:T2 | photon_latency | 10 | 10 | 0 |  |
| single_B5:B5:T3 | baseline | 10 | 9 | -1 | evidence_missing=1 |
| single_C1:C1:T1 | tie | 10 | 10 | 0 |  |
| single_C1:C1:T2 | photon | 7 | 10 | 3 | evidence_missing=1;too_many_citations=8>6 |
| single_C1:C1:T3 | photon | 6 | 10 | 4 | evidence_missing=1;forbidden_cited_paths=7 |
| single_C2:C2:T1 | tie | 10 | 10 | 0 |  |
| single_C2:C2:T2 | photon_latency | 9 | 9 | 0 | evidence_missing=1 / evidence_missing=1 |
| single_C2:C2:T3 | photon_latency | 9 | 9 | 0 | evidence_missing=1 / evidence_missing=1 |
| single_C3:C3:T1 | tie | 10 | 10 | 0 |  |
| single_C3:C3:T2 | tie | 10 | 10 | 0 |  |
| single_C3:C3:T3 | photon | 7 | 10 | 3 | answer_missing=1;evidence_missing=1 |
| single_C4:C4:T1 | tie | 10 | 10 | 0 |  |
| single_C4:C4:T2 | photon | 3 | 10 | 7 | answer_missing=1;evidence_missing=1;forbidden_cited_paths=4;missing_cautious_answer |
| single_C4:C4:T3 | photon | 6 | 10 | 4 | answer_missing=1;evidence_missing=1;forbidden_cited_paths=1 |
