import pandas as pd

# 读取CSV文件
titanic = pd.read_csv('./sqlagent/titanic.csv')


from sqlalchemy import create_engine

#使用SQLAlchemy创建与SQL数据库的连接。这里以SQLite为例，但你也可以连接到MySQL、PostgreSQL等其他数据库：
# 创建数据库引擎
db_path = './sqlagent/titanic.db'
engine = create_engine(f'sqlite:///{db_path}')


# 将数据写入SQL数据库
titanic .to_sql('titanic', con=engine, if_exists='replace', index=False)
