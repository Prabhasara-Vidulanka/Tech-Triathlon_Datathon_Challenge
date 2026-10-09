# AI Tool Disclosure

OpenAI Codex assisted with the Datathon workflow. It helped inspect the supplied documentation and local files, formulate validation and leakage checks, write and execute Python code, compare experiments, diagnose failures, prepare figures and documentation, and assemble submission artifacts.

All competition data processing, label construction, feature generation, model fitting, validation, prediction, and optimization were executed locally through auditable code. The predictive models were trained from scratch on the supplied synthetic competition data. XGBoost used the local NVIDIA GPU; scikit-learn and SciPy handled other local computations. No competition data or derivative was sent to a proprietary modelling API, external website, or third party. No pretrained predictive model, external dataset, AutoML system, or low-code/no-code modelling tool was used.

The team remains responsible for reviewing the label definitions, features, validation design, business priorities, code, final predictions, written explanations, and submission package. AI-generated recommendations were tested against chronological holdouts, rolling backtests, automated assertions, model reload checks, and the official Task 2B validator rather than accepted without verification.
