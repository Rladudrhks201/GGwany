using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
using Microsoft.Azure.Devices.Client;
using Newtonsoft;

namespace IoTClient
{
    internal class Program
    {
        private static System.Timers.Timer SensorTimer;
        private const string DeviceConnectionString = "HostName=labuser26IoThub.azure-devices.net;SharedAccessKeyName=iothubowner;SharedAccessKey=vzCiMmOJOvdOaEOTfAE1ufXHPvPKawoCHuVpZmU1FX4=";
        private const string DeviceID = "Device1";
        private static DeviceClient SensorDevice = null;

        private static DummySensor DummySensor = new DummySensor();
        static void Main(string[] args)
        {
            SetTimer();

            SensorDevice = DeviceClient.CreateFromConnectionString(DeviceConnectionString,DeviceID);

            if(SensorDevice == null)
            {
                Console.WriteLine("Failed to create DeviceClient!");
                SensorTimer.Stop();
            }

            Console.WriteLine("\nPress Enter Key to exit the application...\n");
            Console.WriteLine("The application started at {0:HH:mm:ss.fff}", DateTime.Now);
            Console.ReadLine();
            SensorTimer.Stop();
            SensorTimer.Dispose();  
        }

        private static void SetTimer()
        {
            SensorTimer = new System.Timers.Timer(2000);
            SensorTimer.Elapsed += SensorTimer_Elapsed;
            SensorTimer.AutoReset = true;
            SensorTimer.Enabled = true;
        }

        private static void SensorTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            Console.WriteLine("The Elapsed Event was raised at {0:HH:mm:ss.fff}", e.SignalTime);
            SendEvent();
        }

        private static void SendEvent()
        {
            WeatherModel model = DummySensor.GetWeatherModel(DeviceID);

            string json = Newtonsoft.Json.JsonConvert.SerializeObject(model);

            Console.WriteLine(json);
        }
    }
}
