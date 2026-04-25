from openaq import OpenAQ
import numpy as np

# Initialize the client
# Remember to regenerate your API key in the OpenAQ dashboard!
API_KEY = "9ee18918b15274daeae8843253082fe15c8726b69bf81e567aaa96b2b52981bc"
client = OpenAQ(api_key=API_KEY)

def get_live_data(location_id):
    """
    Fetches data, prints it in your specific format, and RETURNS the dictionary.
    """
    try:
        sensor_metadata = client.locations.sensors(location_id)
        sensor_map = {s.id: {'name': s.parameter.name, 'units': s.parameter.units} for s in sensor_metadata.results}
        latest_data = client.locations.latest(location_id)
        
        measurements = {}
        print(f"--- Data for Location {location_id} ---")
        
        for result in latest_data.results:
            info = sensor_map.get(result.sensors_id, {'name': 'Unknown', 'units': 'N/A'})
            p_name = info['name']
            val = result.value
            unit = info['units']
            
            # Print in your requested format
            print(f"{p_name}: {val} {unit}")
            
            # Store for the simulation
            measurements[p_name] = val
            
        return measurements
    except Exception as e:
        print(f"An error occurred during fetch: {e}")
        return None

def calculate_no2_index(ppb_val):
    """
    Calculates the CPCB NO2 sub-index.
    Converts ppb to ug/m3 first.
    """
    # 1. Convert ppb to ug/m3 (Standard conversion for NO2)
    ug_m3 = ppb_val * 1.88
    
    # 2. CPCB NO2 Breakpoints (24-hour / 1-hour average levels)
    # [Concentration Low, Conc High, Index Low, Index High]
    breakpoints = [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500) # Severe+
    ]
    
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= ug_m3 <= c_high:
            # Linear Interpolation Formula
            index = ((i_high - i_low) / (c_high - c_low)) * (ug_m3 - c_low) + i_low
            return int(index)
    return 500 # Cap at 500

def run_cstr_simulation():
    # --- STEP 1: Fetch Live Data ---
    loc_id = 3409385
    data = get_live_data(loc_id)
    if not data: return

    # --- STEP 2: User Options ---
    print("\n--- Simulation Configuration ---")
    inj_rate = float(input("Enter spray injection rate (suggested 0.05 kg/s): ") or 0.05)
    sim_mins = int(input("Enter simulation duration in minutes (suggested 10): ") or 10)
    
    print("\nSelect Spray Type:")
    print("1. Standard CaCO3 (Basic neutralization)")
    print("2. TiO2/Fe2O3 Coated (High photocatalytic reduction)")
    choice = input("Choice (1 or 2): ")
    
    k_no2 = 0.0004 if choice == "2" else 0.0001
    
    # --- STEP 3: Setup Math ---
    L, W, H = 100, 100, 50 
    Volume = L * W * H
    wind_speed = data.get('wind_speed', 1.0)
    flow_rate = wind_speed * (W * H)
    tau = Volume / flow_rate if flow_rate > 0 else 3600
    
    c_in = data.get('no2', 20.0) 
    c_curr = c_in
    aerosol_mass = 0
    dt = 1
    
    # CALCULATE INITIAL INDEX
    initial_index = calculate_no2_index(c_in)
    
    print(f"\n--- Running CSTR Simulation (Tau: {tau:.1f}s) ---")
    print(f"Initial NO2: {c_in:.2f} ppb | Initial NO2 Index: {initial_index}")
    
    for sec in range(sim_mins * 60):
        aerosol_mass += inj_rate * dt
        aerosol_mass *= 0.99 #calculate from stokes law
        
        dC = ((c_in - c_curr) / tau) - (k_no2 * aerosol_mass * c_curr)
        c_curr += dC * dt
        
        if sec % 60 == 0:
            reduction = ((c_in - c_curr) / c_in) * 100
            print(f"Minute {sec//60}: NO2 = {c_curr:.2f} ppb ({reduction:.1f}% reduction)")

    # CALCULATE FINAL INDEX
    final_index = calculate_no2_index(c_curr)

    print("-" * 30)
    print(f"Final NO2 concentration: {c_curr:.2f} ppb")
    print(f"BEFORE Mitigation NO2 Index: {initial_index}")
    print(f"AFTER Mitigation NO2 Index: {final_index}")
    print(f"Index Improvement: {initial_index - final_index} points")
    print("To get further improvement, please decrease aerosol mist droplet size, or increase injection rate.")

# Execute
run_cstr_simulation()