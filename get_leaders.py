import pandas as pd
from src.data_storage.database_setup import get_engine

engine = get_engine()
query = "SELECT leader, p_value FROM granger_results WHERE target = 'SP500' AND is_significant = TRUE ORDER BY p_value"
df = pd.read_sql(query, engine)
print(df)
