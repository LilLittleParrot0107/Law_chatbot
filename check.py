import requests

url = 'http://127.0.0.1:5000/get_answer'
data = {'question':'Nội dung của chi phí cấu thành chi phí thuê dịch vụ theo yêu cầu là gì?'}

# Convert any sets to lists
data = {key: list(value) if isinstance(value, set) else value for key, value in data.items()}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    result = response.json()
    print(result)
except requests.exceptions.RequestException as e:
    print("Error:", e)