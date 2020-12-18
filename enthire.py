from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config
import pandas as pd
import io
import joblib





joblib_file = "job_lib_nlp_model.pk"
joblib_text_clf = joblib.load(joblib_file)


@view_config(route_name='intern_task', renderer='json')
def getting_json_pred(request):
    text = request.matchdict['airline_text']
    data = pd.read_csv(io.StringIO(text), sep=";")
    prediction = joblib_text_clf.predict(data)
    x = prediction.tolist()
    pred = x[0]
    result = {'associated predicted sentiment in text': pred}
    return result


if __name__ == '__main__':
    with Configurator() as config:

        config.add_route('intern_task', '/{airline_text}')
        config.scan()
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()