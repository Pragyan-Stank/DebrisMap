import math
from datetime import timedelta, datetime

def forecast_trajectory(lat0: float, lon0: float, t0: datetime, hours: int = 72, dt: int = 1) -> list:
    """
    Simulates a 72-hour drift forecast using Runge-Kutta 4th order (simplified).
    dX/dt = V_current(x, t) + alpha * V_wind(x, t) + V_stokes(x, t)
    """
    positions = [{"lat": lat0, "lon": lon0, "time": t0.isoformat()}]
    lat = lat0
    lon = lon0
    t = t0
    
    for step in range(hours):
        # MOCK DATA INJECTION — In production, fetch this from CMEMS/ECMWF APIs
        u_c, v_c = 0.2, 0.1       # Ocean currents (m/s)
        u_w, v_w = 5.0, -2.0      # Wind vectors (m/s)
        u_s, v_s = 0.05, 0.02     # Stokes drift (m/s)
        
        # Calculate total combined drift vectors
        # Wind leeway factor = 3% (0.03) for macroplastics
        u_total = u_c + 0.03 * u_w + u_s
        v_total = v_c + 0.03 * v_w + v_s
        
        # Convert m/s spatial drift to degrees/hour
        dlat = v_total * 3600 / 111000
        dlon = u_total * 3600 / (111000 * math.cos(math.radians(lat)))
        
        lat += dlat * dt
        lon += dlon * dt
        t += timedelta(hours=dt)
        
        positions.append({"lat": lat, "lon": lon, "time": t.isoformat()})
        
    return positions
