import pandas as pd
import re
import math


class DataCleaningEnv:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.step_count = 0
        self.max_steps = 20
        self.current_task = "hard"

    def load_data(self, path: str):
        self.df = pd.read_csv(path)
        self.original_df = self.df.copy()

    def _get_issues(self):
        issues = []
        if self.df is None:
            return issues
        if self.df.duplicated().any():
            issues.append("duplicates")
        if "age" in self.df.columns:
            ages = pd.to_numeric(self.df["age"], errors="coerce")
            if ages.isnull().any() or (ages < 0).any():
                issues.append("missing_age")
        if "email" in self.df.columns:
            email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
            invalid = self.df["email"].apply(
                lambda e: not bool(re.match(email_pattern, str(e))) if pd.notna(e) else False
            )
            if invalid.any():
                issues.append("invalid_email")
        return issues

    def _safe_value(self, v):
        try:
            if math.isnan(v) or math.isinf(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    def _get_observation(self):
        if self.df is not None:
            raw = self.df.head(5).to_dict(orient="records")
            preview = [{k: self._safe_value(val) for k, val in row.items()} for row in raw]
        else:
            preview = []
        return {"data_preview": preview, "issues": self._get_issues()}

    def _get_info(self):
        issues = self._get_issues()
        return {
            "step_count": self.step_count,
            "issues_remaining": len(issues),
            "total_rows": len(self.df) if self.df is not None else 0,
            "task": self.current_task,
        }

    def get_state(self):
        return {
            "observation": self._get_observation(),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_task": self.current_task,
            "issues": self._get_issues(),
            "done": len(self._get_issues()) == 0 or self.step_count >= self.max_steps,
        }

    def reset(self, task: str = "hard"):
        if self.original_df is not None:
            self.df = self.original_df.copy()
        self.step_count = 0
        self.current_task = task
        return self._get_observation()

    def grade(self, task: str) -> float:
        issues = self._get_issues()
        total_issues = 3

        if task == "easy":
            if "duplicates" not in issues:
                return 0.99
            else:
                return 0.01

        elif task == "medium":
            if "missing_age" not in issues:
                return 0.99
            else:
                return 0.01

        elif task == "hard":
            remaining = len(issues)
            if remaining == 0:
                return 0.99
            elif remaining == total_issues:
                return 0.01
            else:
                return round(0.01 + ((total_issues - remaining) / total_issues) * 0.98, 2)

        return 0.01

    def step(self, action: str):
        self.step_count += 1
        reward = -0.1

        if action == "remove_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            if before > len(self.df):
                reward += 0.5

        elif action == "fill_missing":
            if "age" in self.df.columns:
                self.df["age"] = pd.to_numeric(self.df["age"], errors="coerce")
                valid_ages = self.df["age"][(self.df["age"].notna()) & (self.df["age"] >= 0)]
                median_age = valid_ages.median() if len(valid_ages) > 0 else 25
                self.df["age"] = self.df["age"].fillna(median_age)
                self.df.loc[self.df["age"] < 0, "age"] = median_age
                reward += 1.0

        elif action == "fix_email":
            if "email" in self.df.columns:
                email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"

                def clean_email(e):
                    e = str(e).strip()
                    e = re.sub(r"@+", "@", e)
                    if re.match(r"^[\w\.-]+@$", e):
                        e = e + "gmail.com"
                    if "@" not in e:
                        e = e + "@gmail.com"
                    if "@" in e and "." not in e.split("@")[-1]:
                        e = e + ".com"
                    return e

                before_invalid = self.df["email"].apply(
                    lambda e: not bool(re.match(email_pattern, str(e))) if pd.notna(e) else False
                ).sum()
                self.df["email"] = self.df["email"].apply(clean_email)
                after_invalid = self.df["email"].apply(
                    lambda e: not bool(re.match(email_pattern, str(e))) if pd.notna(e) else False
                ).sum()
                if before_invalid > after_invalid:
                    reward += 1.0

        issues = self._get_issues()
        done = len(issues) == 0 or self.step_count >= self.max_steps
        if len(issues) == 0:
            reward += 5.0

        return self._get_observation(), reward, done, self._get_info()

    def get_clean_data(self):
        return self.df