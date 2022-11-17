using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IoTClient
{
    internal class WeatherModel
    {
        public string DeviceID { get; set; }
        public int Temperature { get; set; }
        public int humidity { get; set; }
        public int Dust { get; set; }

    }
}
