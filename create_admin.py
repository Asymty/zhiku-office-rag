import getpass
from contextlib import closing

from pydantic import ValidationError

from access import MIN_PASSWORD, Member, store_member
from app import database


def main():
    with closing(database()) as connection:
        if connection.execute("SELECT 1 FROM members LIMIT 1").fetchone():
            raise SystemExit("首个管理员已创建。请登录网页管理成员。")
        name = input("管理员用户名：").strip()
        try:
            account = Member(name=name, role="admin")
        except ValidationError:
            raise SystemExit("用户名须由 1 至 64 个字母、数字、下划线或短横线组成。")
        password = getpass.getpass(f"管理员密码（至少 {MIN_PASSWORD} 个字符）：")
        if not MIN_PASSWORD <= len(password) <= 128 or password != getpass.getpass("确认密码："):
            raise SystemExit("密码长度无效或两次输入不一致。")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM members LIMIT 1").fetchone():
                raise SystemExit("首个管理员已创建。")
            store_member(connection, account, password)
    print("管理员已创建。请打开知识库页面登录。")


if __name__ == "__main__":
    main()
