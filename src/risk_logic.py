"""Rule-based warning logic: ~300 m road awareness + weather/visibility/speed."""

HAZARD_WEIGHTS = {"pothole": 20, "animal": 30, "pedestrian": 30, "vehicle": 25,
                  "obstacle": 25, "crack": 10, "manhole": 20, "none": 0}


def assess_risk(speed_kmh=40, visibility_m=1000, rainfall_mm=0, hazard="none",
                hazard_distance_m=None, nearby_vehicles=0):
    score, reasons = 0, []

    if speed_kmh > 80:
        score += 25; reasons.append("high speed")
    elif speed_kmh > 60:
        score += 12; reasons.append("fast speed")

    if visibility_m < 100:
        score += 30; reasons.append("very low visibility")
    elif visibility_m < 300:
        score += 15; reasons.append("reduced visibility")

    if rainfall_mm > 10:
        score += 20; reasons.append("heavy rain")
    elif rainfall_mm > 2:
        score += 10; reasons.append("rain")

    h = (hazard or "none").lower()
    if h != "none" and hazard_distance_m is not None and hazard_distance_m <= 300:
        w = HAZARD_WEIGHTS.get(h, 15)
        closeness = 1.0 if hazard_distance_m <= 100 else 0.6
        score += int(w * closeness)
        reasons.append(f"{h} within {int(hazard_distance_m)} m")

    if nearby_vehicles >= 5:
        score += 10; reasons.append("dense traffic")

    score = min(score, 100)
    level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    warning = {"High": "DANGER: slow down now", "Medium": "CAUTION: stay alert", "Low": "Road looks safe"}[level]
    return {"risk_score": score, "risk_level": level, "warning": warning, "reasons": reasons}
