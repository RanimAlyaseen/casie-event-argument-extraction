# Original latency measurements

These are preserved measurements from the fuller original five-model package,
`TeamUltron_Project2 (1).zip`, rather than a new benchmark of this laptop.
The source JSON files are kept unchanged under [results/latency/](../../results/latency/).
The [provenance record](../../results/latency/provenance.json) records archive
and member hashes.

| Supplied model | Reported hardware | Single-event median / p90 | Throughput |
|---|---|---:|---:|
| CRF | CPU | 12.3 / 16.4 ms | 106.0 events/s |
| BERT cased | T4 GPU | 20.6 / 23.9 ms | 143.6 events/s |
| BERT uncased | T4 GPU | 21.4 / 25.1 ms | 144.7 events/s |
| RoBERTa | T4 GPU | 20.7 / 25.2 ms | 145.4 events/s |
| SecureBERT | T4 GPU | 19.6 / 22.1 ms | 146.7 events/s |

The GPU JSON records `cuda:0`; T4 is reported in the accompanying archived
README. The transformer benchmark code uses batch size 1 for single-event
latency, including encoding/tokenization, and batch size 32 for throughput,
excluding encoding. Consequently 20.7 ms and 145.4 events/s are measurements
of different execution paths and are not reciprocals.

CPU and GPU rows use different hardware; they are not a controlled
same-hardware architecture comparison. These artifacts cover the original
five supplied models only. BiLSTM and either DeBERTa method have no matching
latency measurement in this package. A full raw-text demo request may generate
several event queries and additional trigger-processing work.

RoBERTa improves test F1 over the supplied CRF by **9.44 points**, or **10.07
points** over the additional CRF run. Label which baseline is used when
discussing the accuracy/latency tradeoff.
