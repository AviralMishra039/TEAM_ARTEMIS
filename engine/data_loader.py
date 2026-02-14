import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def generate_live_data(
    n_points: int = 300,
    surgery_type: str = "Cardiac",
    seed: Optional[int] = 42
) -> pd.DataFrame:
    try:
        if n_points <= 0:
            raise ValueError("n_points must be positive")
        
        if seed is not None:
            np.random.seed(seed)
        
        time = np.arange(n_points)
        
        # 1. Initialize all as NumPy arrays for consistent length handling
        base_hr = 65 if surgery_type == "Cardiac" else 72
        hr = base_hr + 4 * np.sin(time / 12) + np.random.normal(0, 1.2, n_points)
        
        # Use np.array instead of lists to ensure fixed length
        activity = np.array(["Resting"] * n_points, dtype=object)
        meds = np.array(["Beta-Blockers"] * n_points, dtype=object)
        surgery_types = np.array([surgery_type] * n_points, dtype=object)
        
        # 2. Realistic activity variations
        for i in range(50, n_points, 50):
            if i + 10 < n_points:
                activity[i:i+10] = "Walking"  # NumPy broadcasting handles this perfectly
                hr[i:i+10] += np.random.normal(15, 3, 10)
        
        # 3. ANOMALIES (Exact indexing)
        # Point Spike
        if n_points > 50:
            hr[50] += 30
        
        # Drift
        if n_points > 200:
            hr[150:200] += np.linspace(0, 20, 50)
        
        # Unstable Pattern
        if n_points > 280:
            hr[250:280] = 80 + np.random.normal(0, 8, 30)
            activity[250:280] = "Resting"
        
        # 4. Final Processing
        hr = np.clip(hr, 40, 200)
        
        # Since they are all NumPy arrays of length n_points, this will NOT fail
        df = pd.DataFrame({
            "timestamp": time,
            "hr": np.round(hr, 2),
            "activity": activity,
            "medication": meds,
            "surgery_type": surgery_types
        })
        
        logger.info(f"Generated {n_points} data points for {surgery_type}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating live data: {str(e)}")
        raise