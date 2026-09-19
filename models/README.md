# Fitted model bundles

The repository includes the six fitted models used by external validation.

| File | SHA-256 |
| --- | --- |
| `final_pl_peak_expensive.pkl` | `ad11abefb5edbfbee6d37e5e8ccf1dea688dca05dc35de3e93660d0d2bf411e0` |
| `final_pl_peak_lowcost.pkl` | `682f71f19e89bf4e8ba4cef074f2fe8af911e123488454515249bf7c4cf9c4e2` |
| `final_plqy_rf_expensive.pkl` | `8355c1c3ed869adcb64e0ec2a5ef40023dac797e729837dadef1015a67cf56a5` |
| `final_plqy_rf_lowcost.pkl` | `998989e61899a40537115106d110049d665d34a611572115b67723fd6971038e` |
| `final_fwhm_rf_expensive.pkl` | `181847b4b625bbe3bb3e33e6f039f1928ad3184313be164963fdbbd26dfe64a0` |
| `final_fwhm_rf_lowcost.pkl` | `cfb708f102b968215a61aae6f6e637d37e77573a727aaf858a2bd0a4eaab2c55` |

Most files are dictionaries containing an estimator, ordered feature names,
and relevant metadata. The augmented PLQY file is a fitted scikit-learn
pipeline and exposes its feature order through the estimator. The external
validation script supports both formats.

The inspected scikit-learn objects were serialized with scikit-learn 1.7.2,
which is recorded in `requirements.txt`.

Python pickle files can execute code while loading. Only load these files from
a trusted source and verify their checksums after transfer.
