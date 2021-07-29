from rule_estimator import *
from rule_estimator.datasets import *

X, y = titanic_X_y()

db = RuleClassifierDashboard(X, y, val_size=0.25, labels=titanic_labels)

app = db.app.server