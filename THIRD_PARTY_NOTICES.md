# Third-party notices and license scope

The root [MIT License](LICENSE) applies to original code and documentation
contributed to this project, to the extent the contributors hold the rights
to that material. It does not grant rights in material owned by third parties
or override another copyright or license notice.

## CASIE corpus and input bundle

The input bundle, `data/TeamUltron_Project2_Inputs.zip`, includes CASIE-derived
news text, annotations, corrected data representations and supplied prediction
artifacts. **The input bundle is not covered by this project's MIT license.**
Adding the code license does not establish or expand permission to redistribute
the corpus, articles, annotations or supplied prediction artifacts. Consult
their respective rights holders and applicable source terms for those rights.
The existing input bundle is retained unchanged by this licensing update.

Corpus source: [Ebiquity/CASIE](https://github.com/Ebiquity/CASIE).
The upstream project requests citation of its research. Please cite:

> Taneeya Satyapanich, Francis Ferraro and Tim Finin. CASIE: Extracting
> Cybersecurity Event Information from Text. AAAI, 2020.
> [DOI: 10.1609/aaai.v34i05.6401](https://doi.org/10.1609/aaai.v34i05.6401).

See [dataset preparation](DATASET_PREPARATION.md) for the corrected split,
alignment details and input provenance. Public availability and academic
citation do not by themselves assign ownership to this project's contributors.

## Models, checkpoints and dependencies

Pretrained models, tokenizers, separately shared fine-tuned checkpoints and
third-party software packages retain their own applicable terms. This project's
MIT license does not relicense them. Checkpoint weights are not committed to
this repository. Consult the original model distributions and the packages
listed in [requirements.txt](requirements.txt) for their license information.

## Embedded content and research references

Third-party code, article excerpts, annotations, data-derived prediction
artifacts or other third-party content embedded in notebooks and reports
retain their original terms. Their inclusion does not place them under MIT.
Preserve existing notices when copying or adapting that content.

The benchmark documentation also cites
[TextEE](https://aclanthology.org/2024.findings-acl.760/) and
[TAGPRIME](https://aclanthology.org/2023.acl-long.723/). These references
acknowledge related work; the repository license does not extend to those
papers or their authors' independent implementations.
