"""
Customer Churn Defuser - Synthetic Dataset Generator
Generates a realistic dataset of 10,000 customer records with churn labels
driven by sensible business logic (low engagement, support issues, payment
failures, inactivity -> higher churn probability).
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
NUM_RECORDS = 10000

np.random.seed(RANDOM_SEED)


def generate_dataset(n=NUM_RECORDS):
    customer_id = np.arange(100000, 100000 + n)

    age = np.random.randint(18, 70, size=n)
    gender = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])
    subscription_type = np.random.choice(
        ["Basic", "Standard", "Premium"], size=n, p=[0.4, 0.4, 0.2]
    )

    base_fee = {"Basic": 9.99, "Standard": 19.99, "Premium": 39.99}
    monthly_fee = np.array([base_fee[s] for s in subscription_type]) + np.round(
        np.random.normal(0, 2, size=n), 2
    )
    monthly_fee = np.clip(monthly_fee, 5, 100)

    subscription_months = np.random.randint(1, 60, size=n)
    login_frequency = np.random.poisson(lam=8, size=n)
    login_frequency = np.clip(login_frequency, 0, 60)

    session_duration = np.round(np.random.gamma(shape=2.0, scale=10.0, size=n), 2)
    session_duration = np.clip(session_duration, 1, 180)

    support_tickets = np.random.poisson(lam=1.2, size=n)
    support_tickets = np.clip(support_tickets, 0, 15)

    payment_failures = np.random.poisson(lam=0.4, size=n)
    payment_failures = np.clip(payment_failures, 0, 10)

    last_active_days = np.random.exponential(scale=15, size=n).astype(int)
    last_active_days = np.clip(last_active_days, 0, 180)

    customer_satisfaction = np.clip(
        np.round(np.random.normal(7, 1.8, size=n), 1), 1, 10
    )

    feature_usage_score = np.clip(
        np.round(np.random.normal(55, 20, size=n), 1), 0, 100
    )

    referrals = np.random.poisson(lam=0.6, size=n)
    referrals = np.clip(referrals, 0, 10)

    engagement_score = np.clip(
        (
            0.3 * (login_frequency / 60 * 100)
            + 0.3 * feature_usage_score
            + 0.2 * (session_duration / 180 * 100)
            + 0.2 * (customer_satisfaction / 10 * 100)
        ),
        0,
        100,
    )
    engagement_score = np.round(engagement_score, 1)

    # ---- Business-logic driven churn probability ----
    risk_score = (
        0.25 * (1 - login_frequency / max(login_frequency.max(), 1))
        + 0.20 * (support_tickets / max(support_tickets.max(), 1))
        + 0.20 * (1 - engagement_score / 100)
        + 0.20 * (last_active_days / max(last_active_days.max(), 1))
        + 0.15 * (payment_failures / max(payment_failures.max(), 1))
    )

    # Normalize risk_score to 0-1 then convert to churn probability with some noise
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    churn_prob = np.clip(risk_score + np.random.normal(0, 0.07, size=n), 0, 1)

    churn = (churn_prob > 0.55).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "subscription_type": subscription_type,
            "monthly_fee": monthly_fee,
            "subscription_months": subscription_months,
            "login_frequency": login_frequency,
            "session_duration": session_duration,
            "support_tickets": support_tickets,
            "payment_failures": payment_failures,
            "last_active_days": last_active_days,
            "customer_satisfaction": customer_satisfaction,
            "feature_usage_score": feature_usage_score,
            "referrals": referrals,
            "engagement_score": engagement_score,
            "churn": churn,
        }
    )

    return df


def main():
    df = generate_dataset()
    os.makedirs("dataset", exist_ok=True)
    out_path = os.path.join("dataset", "customer_churn_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset generated successfully: {out_path}")
    print(f"Total records: {len(df)}")
    print(f"Churn rate: {df['churn'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
