# common/tep_variables.py - TEP Dataset Variable Descriptions

TEP_VARIABLES = {
    1: {
        "name": "A Feed (stream 1)",
        "description": "Feed flow rate of component A",
        "unit": "kscmh",
        "type": "flow_rate",
        "typical_range": "0.25-0.35",
        "scaling_priority": "high"
    },
    2: {
        "name": "D Feed (stream 2)", 
        "description": "Feed flow rate of component D",
        "unit": "kg/h",
        "type": "flow_rate",
        "typical_range": "3000-4000",
        "scaling_priority": "high"
    },
    3: {
        "name": "E Feed (stream 3)",
        "description": "Feed flow rate of component E", 
        "unit": "kg/h",
        "type": "flow_rate",
        "typical_range": "8000-9000",
        "scaling_priority": "high"
    },
    4: {
        "name": "A and C Feed (stream 4)",
        "description": "Total feed flow rate of A and C",
        "unit": "kscmh", 
        "type": "flow_rate",
        "typical_range": "0.1-0.3",
        "scaling_priority": "high"
    },
    5: {
        "name": "Compressor Recycle Valve",
        "description": "Recycle flow from compressor",
        "unit": "kscmh",
        "type": "flow_rate", 
        "typical_range": "24-28",
        "scaling_priority": "medium"
    },
    6: {
        "name": "Reactor Feed Rate",
        "description": "Total feed rate to reactor",
        "unit": "kscmh",
        "type": "flow_rate",
        "typical_range": "40-45",
        "scaling_priority": "high"
    },
    7: {
        "name": "Reactor Pressure",
        "description": "Operating pressure in reactor",
        "unit": "kPa gauge",
        "type": "pressure",
        "typical_range": "2700-2800",
        "scaling_priority": "critical"
    },
    8: {
        "name": "Reactor Level",
        "description": "Liquid level in reactor",
        "unit": "%",
        "type": "level",
        "typical_range": "65-75",
        "scaling_priority": "critical"
    },
    9: {
        "name": "Reactor Temperature",
        "description": "Operating temperature in reactor",
        "unit": "°C",
        "type": "temperature",
        "typical_range": "120-125",
        "scaling_priority": "critical"
    },
    10: {
        "name": "Purge Rate (stream 9)",
        "description": "Purge flow rate",
        "unit": "kscmh",
        "type": "flow_rate",
        "typical_range": "0.1-0.5",
        "scaling_priority": "medium"
    },
    11: {
        "name": "Product Sep Temp",
        "description": "Temperature in product separator",
        "unit": "°C", 
        "type": "temperature",
        "typical_range": "80-85",
        "scaling_priority": "high"
    },
    12: {
        "name": "Product Sep Level",
        "description": "Level in product separator",
        "unit": "%",
        "type": "level", 
        "typical_range": "30-35",
        "scaling_priority": "high"
    },
    13: {
        "name": "Product Sep Pressure",
        "description": "Pressure in product separator",
        "unit": "kPa gauge",
        "type": "pressure",
        "typical_range": "2650-2700",
        "scaling_priority": "high"
    },
    14: {
        "name": "Product Sep Underflow",
        "description": "Underflow from product separator",
        "unit": "m3/h",
        "type": "flow_rate",
        "typical_range": "18-24",
        "scaling_priority": "medium"
    },
    15: {
        "name": "Stripper Level",
        "description": "Level in stripper",
        "unit": "%",
        "type": "level",
        "typical_range": "47-52",
        "scaling_priority": "high"
    },
    16: {
        "name": "Stripper Pressure",
        "description": "Pressure in stripper",
        "unit": "kPa gauge", 
        "type": "pressure",
        "typical_range": "3100-3200",
        "scaling_priority": "high"
    },
    17: {
        "name": "Stripper Underflow",
        "description": "Underflow from stripper",
        "unit": "m3/h",
        "type": "flow_rate",
        "typical_range": "22-24",
        "scaling_priority": "medium"
    },
    18: {
        "name": "Stripper Temperature",
        "description": "Temperature in stripper",
        "unit": "°C",
        "type": "temperature", 
        "typical_range": "65-70",
        "scaling_priority": "high"
    },
    19: {
        "name": "Stripper Steam Flow",
        "description": "Steam flow to stripper",
        "unit": "kg/h",
        "type": "flow_rate",
        "typical_range": "230-250",
        "scaling_priority": "medium"
    },
    20: {
        "name": "Compressor Work",
        "description": "Work performed by compressor",
        "unit": "kW",
        "type": "power",
        "typical_range": "340-360",
        "scaling_priority": "medium"
    },
    21: {
        "name": "Reactor Cooling Water Outlet Temp",
        "description": "Cooling water outlet temperature",
        "unit": "°C",
        "type": "temperature",
        "typical_range": "35-40",
        "scaling_priority": "medium"
    },
    22: {
        "name": "Separator Cooling Water Outlet Temp", 
        "description": "Separator cooling water outlet temperature",
        "unit": "°C",
        "type": "temperature",
        "typical_range": "40-45",
        "scaling_priority": "medium"
    },
    # Composition variables (23-41)
    23: {
        "name": "Reactor Feed A",
        "description": "Component A composition in reactor feed",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "32-35",
        "scaling_priority": "critical"
    },
    24: {
        "name": "Reactor Feed B", 
        "description": "Component B composition in reactor feed",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "13-15",
        "scaling_priority": "critical"
    },
    25: {
        "name": "Reactor Feed C",
        "description": "Component C composition in reactor feed", 
        "unit": "mol%",
        "type": "composition",
        "typical_range": "24-26",
        "scaling_priority": "critical"
    },
    26: {
        "name": "Reactor Feed D",
        "description": "Component D composition in reactor feed",
        "unit": "mol%", 
        "type": "composition",
        "typical_range": "13-15",
        "scaling_priority": "critical"
    },
    27: {
        "name": "Reactor Feed E",
        "description": "Component E composition in reactor feed",
        "unit": "mol%",
        "type": "composition", 
        "typical_range": "13-15",
        "scaling_priority": "critical"
    },
    28: {
        "name": "Reactor Feed F",
        "description": "Component F composition in reactor feed",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "0.1-0.3",
        "scaling_priority": "critical"
    },
    29: {
        "name": "Purge A",
        "description": "Component A composition in purge",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "23-25",
        "scaling_priority": "high"
    },
    30: {
        "name": "Purge B",
        "description": "Component B composition in purge", 
        "unit": "mol%",
        "type": "composition",
        "typical_range": "15-17",
        "scaling_priority": "high"
    },
    31: {
        "name": "Purge C",
        "description": "Component C composition in purge",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "24-26", 
        "scaling_priority": "high"
    },
    32: {
        "name": "Purge D",
        "description": "Component D composition in purge",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "11-13",
        "scaling_priority": "high"
    },
    33: {
        "name": "Purge E", 
        "description": "Component E composition in purge",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "12-14",
        "scaling_priority": "high"
    },
    34: {
        "name": "Purge F",
        "description": "Component F composition in purge",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "11-13",
        "scaling_priority": "high"
    },
    35: {
        "name": "Purge G",
        "description": "Component G composition in purge",
        "unit": "mol%", 
        "type": "composition",
        "typical_range": "0.1-0.3",
        "scaling_priority": "high"
    },
    36: {
        "name": "Purge H",
        "description": "Component H composition in purge",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "0.8-1.0",
        "scaling_priority": "high"
    },
    37: {
        "name": "Product D",
        "description": "Component D composition in product",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "53-55",
        "scaling_priority": "critical"
    },
    38: {
        "name": "Product E",
        "description": "Component E composition in product", 
        "unit": "mol%",
        "type": "composition",
        "typical_range": "43-45",
        "scaling_priority": "critical"
    },
    39: {
        "name": "Product F",
        "description": "Component F composition in product",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "0.8-1.0",
        "scaling_priority": "critical"
    },
    40: {
        "name": "Product G",
        "description": "Component G composition in product",
        "unit": "mol%",
        "type": "composition", 
        "typical_range": "0.4-0.6",
        "scaling_priority": "critical"
    },
    41: {
        "name": "Product H",
        "description": "Component H composition in product",
        "unit": "mol%",
        "type": "composition",
        "typical_range": "0.01-0.05",
        "scaling_priority": "critical"
    },
    # Manipulated variables (42-52)
    42: {
        "name": "D Feed Flow (MV)",
        "description": "Manipulated variable - D feed flow",
        "unit": "kg/h",
        "type": "manipulated_variable",
        "typical_range": "3000-4000",
        "scaling_priority": "critical"
    },
    43: {
        "name": "E Feed Flow (MV)",
        "description": "Manipulated variable - E feed flow",
        "unit": "kg/h", 
        "type": "manipulated_variable",
        "typical_range": "8000-9000",
        "scaling_priority": "critical"
    },
    44: {
        "name": "A Feed Flow (MV)",
        "description": "Manipulated variable - A feed flow",
        "unit": "kscmh",
        "type": "manipulated_variable",
        "typical_range": "0.25-0.35",
        "scaling_priority": "critical"
    },
    45: {
        "name": "A and C Feed Flow (MV)",
        "description": "Manipulated variable - A and C feed flow",
        "unit": "kscmh",
        "type": "manipulated_variable", 
        "typical_range": "0.1-0.3",
        "scaling_priority": "critical"
    },
    46: {
        "name": "Compressor Recycle Valve (MV)",
        "description": "Manipulated variable - Compressor recycle valve",
        "unit": "%",
        "type": "manipulated_variable",
        "typical_range": "20-25",
        "scaling_priority": "critical"
    },
    47: {
        "name": "Purge Valve (MV)",
        "description": "Manipulated variable - Purge valve",
        "unit": "%", 
        "type": "manipulated_variable",
        "typical_range": "40-45",
        "scaling_priority": "critical"
    },
    48: {
        "name": "Separator Pot Liquid Flow (MV)",
        "description": "Manipulated variable - Separator pot liquid flow",
        "unit": "%",
        "type": "manipulated_variable",
        "typical_range": "35-40",
        "scaling_priority": "critical"
    },
    49: {
        "name": "Stripper Liquid Product Flow (MV)",
        "description": "Manipulated variable - Stripper liquid product flow",
        "unit": "%",
        "type": "manipulated_variable",
        "typical_range": "30-35",
        "scaling_priority": "critical"
    },
    50: {
        "name": "Stripper Steam Valve (MV)",
        "description": "Manipulated variable - Stripper steam valve",
        "unit": "%",
        "type": "manipulated_variable",
        "typical_range": "47-52",
        "scaling_priority": "critical"
    },
    51: {
        "name": "Reactor Cooling Water Flow (MV)",
        "description": "Manipulated variable - Reactor cooling water flow",
        "unit": "%",
        "type": "manipulated_variable",
        "typical_range": "40-45",
        "scaling_priority": "critical"
    },
    52: {
        "name": "Condenser Cooling Water Flow (MV)",
        "description": "Manipulated variable - Condenser cooling water flow",
        "unit": "%",
        "type": "manipulated_variable",
        "typical_range": "35-40",
        "scaling_priority": "critical"
    }
}

def get_variables_by_type(var_type):
    """Get all variables of a specific type"""
    return {k: v for k, v in TEP_VARIABLES.items() if v["type"] == var_type}

def get_variables_by_priority(priority):
    """Get all variables with specific scaling priority"""
    return {k: v for k, v in TEP_VARIABLES.items() if v["scaling_priority"] == priority}

def get_variable_info(var_id):
    """Get information about a specific variable"""
    return TEP_VARIABLES.get(var_id, None)

