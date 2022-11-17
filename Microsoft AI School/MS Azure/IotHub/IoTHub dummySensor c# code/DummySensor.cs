using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IoTClient
{
    internal class DummySensor
    {
        private Random _Random = new Random();

        public WeatherModel GetWeatherModel(string deviceID)
        {
            var model = new WeatherModel();
            
            model.DeviceID = deviceID;
            model.Temperature = _Random.Next(20,35);
            model.humidity = _Random.Next(40, 80);
            model.Dust = model.Temperature + 50 + _Random.Next(1, 5);

            return model;
        }
    }
}
