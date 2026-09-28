"""심부전 환자 생존 예측 - 데이터 누수(time 변수) 점검 및 재평가"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# 한글 폰트: 설치된 폰트 중 첫 번째 사용 (Windows: Malgun Gothic, macOS: AppleGothic)
_available = {f.name for f in font_manager.fontManager.ttflist}
for _name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if _name in _available:
        plt.rc("font", family=_name)
        break
plt.rc("axes", unicode_minus=False)
os.makedirs("images", exist_ok=True)

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
y = df["DEATH_EVENT"]
X_all = df.drop(columns="DEATH_EVENT")
X_nt = X_all.drop(columns="time")
print(f"환자 수: {len(df)}, 사망 비율: {y.mean():.1%}")

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

rows = []
for setting, X in [("time 포함", X_all), ("time 제외", X_nt)]:
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=scoring)
        row = {"설정": setting, "모델": name}
        for s in scoring:
            row[s] = f"{r['test_'+s].mean():.3f} ± {r['test_'+s].std():.3f}"
            row[s + "_mean"] = r["test_" + s].mean()
        rows.append(row)
res = pd.DataFrame(rows)
print(res[["설정", "모델"] + scoring].to_string(index=False))
res.to_csv("results.csv", index=False, encoding="utf-8-sig")

# 그림 1: time 포함/제외 ROC-AUC 비교
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(models)
w = 0.38; x = np.arange(len(names))
a = [res[(res["설정"] == "time 포함") & (res["모델"] == n)]["roc_auc_mean"].item() for n in names]
b = [res[(res["설정"] == "time 제외") & (res["모델"] == n)]["roc_auc_mean"].item() for n in names]
ax.bar(x - w/2, a, w, label="time 포함 (누수)", color="#bbbbbb")
ax.bar(x + w/2, b, w, label="time 제외 (수정)", color="#d62728")
for i in range(len(names)):
    ax.text(x[i]-w/2, a[i]+0.01, f"{a[i]:.2f}", ha="center", fontsize=9)
    ax.text(x[i]+w/2, b[i]+0.01, f"{b[i]:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("ROC-AUC (5-fold 평균, 0.5 = 무작위 수준)")
ax.set_title("추적 관찰 기간(time) 포함 여부에 따른 성능 비교")
ax.legend(); plt.tight_layout(); plt.savefig("images/auc_with_without_time.png", dpi=150); plt.close()

# 그림 2: 왜 time이 누수인가 - 생존/사망별 관찰 기간 분포
fig, ax = plt.subplots(figsize=(7, 4.5))
lab = y.map({0: "생존", 1: "사망"})
data = [df.loc[lab == "생존", "time"], df.loc[lab == "사망", "time"]]
bp = ax.boxplot(data, patch_artist=True, widths=0.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(["생존", "사망"])
for p, c in zip(bp["boxes"], ["#1f77b4", "#d62728"]): p.set_facecolor(c); p.set_alpha(0.6)
ax.set_ylabel("추적 관찰 기간 (일)")
ax.set_title("사망 환자는 관찰이 일찍 끝나 time이 짧다 → 결과를 반영한 변수")
plt.tight_layout(); plt.savefig("images/time_by_outcome.png", dpi=150); plt.close()

# 그림 3: time 제외 Random Forest의 순열 중요도 (교차검증, 검증 fold에서 ROC-AUC 감소량)
from sklearn.inspection import permutation_importance
perm = []
for tr, te in cv.split(X_nt, y):
    m = models["Random Forest"].fit(X_nt.iloc[tr], y.iloc[tr])
    r = permutation_importance(m, X_nt.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=20, random_state=0)
    perm.append(r.importances_mean)
perm = pd.DataFrame(perm, columns=X_nt.columns)
imp = perm.mean().sort_values()
err = perm.std()[imp.index]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index, imp.values, xerr=err.values, color="#d62728", alpha=0.8, ecolor="gray", capsize=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("변수를 섞었을 때 ROC-AUC 감소량 (5-fold 평균, 오차막대 = 표준편차)")
ax.set_title("time 제외 Random Forest의 순열 중요도")
plt.tight_layout(); plt.savefig("images/feature_importance_no_time.png", dpi=150); plt.close()
print(imp.sort_values(ascending=False).round(4).to_string())

# 그림 4: ROC 곡선 (time 제외, 교차검증 예측)
fig, ax = plt.subplots(figsize=(6, 6))
for name, m in models.items():
    p = cross_val_predict(m, X_nt, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, label=f"{name} (합산 AUC {roc_auc_score(y, p):.2f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("위양성률 (FPR)"); ax.set_ylabel("재현율 (TPR)")
ax.set_title("ROC 곡선 (time 제외, 5-fold 검증 예측을 합쳐 계산)"); ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("images/roc_no_time.png", dpi=150); plt.close()
