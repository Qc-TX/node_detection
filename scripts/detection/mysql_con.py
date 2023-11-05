import pymysql


class MysqlTool:
    def __init__(self):
        self.host = 'localhost'
        self.port = 3306
        self.user = 'root'
        self.password = '12345678'
        self.mysql_conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            passwd=self.password,
            database='aptsys'
#            auth_plugin='mysql_native_password'
        )


    def execute(self, sql: str, args: tuple = None, commit: bool = False) -> any:
        """执行 SQL 语句"""
        try:
            with self.mysql_conn.cursor() as cursor:
                cursor.execute(sql, args)
                if commit:
                    self.mysql_conn.commit()
                    print(f"执行 SQL 语句：{sql}，参数：{args}，数据提交成功")
                else:
                    result = cursor.fetchall()
                    print(f"执行 SQL 语句：{sql}，参数：{args}，查询到的数据为：{result}")
                    return result
        except Exception as e:
            print(f"执行 SQL 语句出错：{e}")
#            self.mysql_conn.rollback()
            raise e


def write_apt_type(apt_type: int, threat_id: str):
    db = MysqlTool()
    sql = "update threat set apt_type=%s where threat_id=%s"
    args = (apt_type, threat_id)
    db.execute(sql, args, commit=True)


def write_node_result(json_node):
    db = MysqlTool()
    sql = "insert into node_detection values (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    args = (json_node['id'], json_node['uuid'], json_node['type'], json_node['pred_type'], json_node['time'],
            json_node['wrong'], json_node['threat_id'], json_node['apt_type'], json_node['ip'])
    db.execute(sql, args, commit=True)