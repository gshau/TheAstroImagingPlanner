import requests
import json

import pandas as pd


def query_status(target_name, port, only_ok=True):
    response = requests.get(f"http://localhost:{port}/status/{target_name}")
    if response.status_code == 200:
        data = response.content
        df = pd.DataFrame(json.loads(data))
        if df.shape[0] > 0:
            if only_ok:
                df = df[df["is_ok"].astype(bool)]
        return df
