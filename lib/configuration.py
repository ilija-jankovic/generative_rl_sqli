

from typing import Dict


class Configuration:

    open_url: str
    headers: Dict[str, str]
    cookie: str


    @property
    def headers_with_cookie(self):
        headers = self.headers.copy()
        headers.update({'cookie': self.cookie})
        
        return headers


    def __init__(
        self,
        open_url: str,
        headers: Dict[str, str],
        cookie: str,
    ) -> None:
        self.open_url = open_url
        self.headers = headers
        self.cookie = cookie