
import requests

URL = "https://translate.googleapis.com/translate_a/single"

text = "В одном селе жил когда-то пёс."

params = {
    "client": "gtx",
    "sl": "ru",
    "tl": "av",
    "dt": "t",
    "q": text,
}

try:
    response = requests.get(
        URL,
        params=params,
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    translated = "".join(
        part[0]
        for part in data[0]
        if part and part[0]
    )

    print("Русский:", text)
    print("Аварский:", translated)

except Exception as e:
    print("Ошибка:", repr(e))
