"""
Test case class for testing Flask application routes.
"""

import unittest

from app import app


class AppTestCase(unittest.TestCase):
    """
    This class contains unit tests for the '/reload' and '/predict' routes
    of the Flask application defined in app.py. The tests ensure that the
    routes return the expected HTTP status codes and response content.
    """

    def setUp(self):
        """
        Set up the test client before each test.

        This method initializes the Flask test client, allowing us to send
        HTTP requests to the application in a controlled test environment.
        It also sets the testing mode to True, which enables certain features
        like better error messages during tests.
        """
        self.app = app.test_client()
        self.app.testing = True

    def test_01_reload_route(self):
        """
        Test the /reload route.

        This test sends a POST request to the /reload route and verifies
        that the response status code is 200 (OK) and that the response
        JSON contains a 'reloaded' key.
        """
        response = self.app.post('/reload')
        self.assertEqual(response.status_code, 200)
        self.assertIn('reloaded', response.get_json())

    def test_02_predict_route(self):
        """
        Test the /predict route.

        This test sends a POST request to the /predict route with a JSON
        payload containing test data. It verifies that the response status
        code is 200 (OK) and that the response JSON contains a 'prediction' key.
        """
        test_data = {
            'data': [
                {
                    'VendorID': 2,
                    'pickup_datetime': '2024-01-01 00:57:55',
                    'dropoff_datetime': '2024-01-01 01:17:43',
                    'passenger_count': 1.0,
                    'trip_distance': 1.72,
                    'RatecodeID': 1.0,
                    'store_and_fwd_flag': 'N',
                    'PULocationID': 186,
                    'DOLocationID': 79,
                    'payment_type': 2,
                    'fare_amount': 17.7,
                    'extra': 1.0,
                    'mta_tax': 0.5,
                    'tip_amount': 0.0,
                    'tolls_amount': 0.0,
                    'improvement_surcharge': 1.0,
                    'total_amount': 22.7,
                    'congestion_surcharge': 2.5,
                    'Airport_fee': 0.0,
                    'execution_date': '2024-01',
                    'type_tripdata': 'yellow',
                }
            ]
        }
        response = self.app.post('/predict', json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('predictions', response.get_json())


if __name__ == '__main__':
    unittest.main()
