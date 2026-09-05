### Build Challenges & Technical Obstacles

The most significant challenges during this build stemmed from enforcing mathematical rigor and preventing data leakage across a multi-model pipeline. 

**Preventing Stacking Leakage via Out-Of-Fold (OOF) Predictions**
When building the meta-learner layer, initial attempts trained the Logistic Regression ensemble on the same data used by the base models. This caused massive overfitting, as the meta-learner learned to over-trust Random Forest (which easily memorizes training data) and failed entirely on the test set. I solved this by implementing a strict 5-fold cross-validated Out-Of-Fold (OOF) generation script. The base models were forced to predict on data they hadn't seen during training, ensuring the meta-learner only evaluated the *generalized* capabilities of its inputs.

**Rejecting the Sunk Cost of Underperforming Stacks**
A major conceptual hurdle was realizing that more complex architectures don't uniformly improve performance across all assets. For `LT.NS` classification, the stack successfully boosted accuracy by +1.09% over the best base model. But for `LT.NS` regression, and across both tracks for `TCS.NS` and `RELIANCE.NS`, the meta-learner failed to generalize better than the standalone models (e.g., the TCS regression stack improved RMSE by a negligible 0.25%). Rather than forcing the complex ensemble into production to validate the engineering effort, I solved this by implementing a dynamic routing architecture. The system reads a `model_selection.json` configuration at inference time, strictly dropping the stack in favor of a single tuned XGBoost or LSTM model unless the ensemble clears a defined 0.5% margin.

**Schema Desyncs Between Backend and Frontend**
As the risk-scoring formulas evolved (particularly fixing the VaR Z-score to 1.645 and decoupling the regime gate's thresholds), the backend API payload shape drifted from what the Streamlit dashboard expected. Initially, this caused silent UI failures where the dashboard rendered `0.0` or misaligned values rather than crashing. I solved this by replacing manual "does it crash" validation with strict `pytest` fixtures. I wrote tests that forcefully read the frontend UI keys and validated them against the real backend dictionary outputs, turning silent UI bugs into immediate pipeline failures.
