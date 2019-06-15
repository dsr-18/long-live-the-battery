import googleapiclient.discovery

instances = [
  [6.7, 3.1, 4.7, 1.5],
  [4.6, 3.1, 1.5, 0.2],
]

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}/versions/{}'.format('ion_age', 'ion_age', 'v0004')

response = service.projects().predict(
    name=name,
    body={'instances': instances}
).execute()

if 'error' in response:
    raise RuntimeError(response['error'])
else:
  print(response['predictions'])