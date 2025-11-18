import psutil
import time
import csv

# Function to get real-time system data
def get_system_data():
    # CPU usage in percentage
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # Memory usage in percentage
    memory_info = psutil.virtual_memory().percent
    
    # Battery status (if available)
    battery = psutil.sensors_battery()
    if battery:
        battery_percent = battery.percent
        power_plugged = battery.power_plugged
    else:
        battery_percent = None
        power_plugged = None

    return cpu_usage, memory_info, battery_percent, power_plugged

# Function to save the collected data to a CSV file
def save_data_to_csv(data):
    with open("energy_data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Function to continuously monitor the system's energy usage and save the data
def monitor_energy():
    # Write the header to the CSV file (only done once)
    with open("energy_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU Usage (%)", "Memory Usage (%)", "Battery (%)", "Power Plugged"])

    while True:
        cpu_usage, memory_info, battery_percent, power_plugged = get_system_data()
        
        # Get the current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare data to save
        data = [timestamp, cpu_usage, memory_info, battery_percent, power_plugged]
        save_data_to_csv(data)

        # Print real-time stats
        print(f"{timestamp} | CPU: {cpu_usage}% | Memory: {memory_info}% | Battery: {battery_percent}% | Plugged In: {'Yes' if power_plugged else 'No'}")

        # Pause for 5 seconds before collecting data again
        time.sleep(5)

if __name__ == "__main__":
    monitor_energy()
