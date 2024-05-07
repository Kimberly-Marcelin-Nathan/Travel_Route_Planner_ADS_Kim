import math
import random
import streamlit as st
import openai
import re
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
import requests
import networkx as nx
import matplotlib.pyplot as plt

from config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY



def generate_distance(locations):
    # Generate a prompt for GPT-3.5
    prompt = "Calculate the total distance between the following locations:\n"
    for i in range(len(locations) - 1):
        for j in range(i + 1, len(locations)):
            prompt += f"{locations[i]} {locations[j]}\n"

    # Call GPT-3.5 to generate the distances
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()


st.set_page_config(
    page_title='Travel Route Planner', page_icon='content\icon.png')

def knapsack_max_value(weights, values, W):
    """
    Calculate the maximum value for the knapsack problem and return the DP table.

    :param weights: List of weights of the items
    :param values: List of values of the items
    :param W: Capacity of the knapsack
    :return: Maximum value and the DP table
    """
    n = len(weights)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    max_value = dp[n][W]
    return max_value, dp

def find_included_items(weights, dp):
    """
    Find the items included in the optimal solution based on the DP table.

    :param weights: List of weights of the items
    :param dp: DP table computed by knapsack_max_value
    :return: List of included items (1-based index)
    """
    n = len(weights)
    W = len(dp[0]) - 1
    included_items = []
    i, w = n, W

    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            included_items.append(i)
            w -= weights[i - 1]
        i -= 1

    included_items.reverse()
    return included_items

def display_route(location_route, x, locations, loc_df, distance_matrix):
    num_locations = len(locations)
    route = [0]
    current_place = 0

    location_route_with_coordinates = []
    for loc in location_route:
        if isinstance(loc, str):
            location = loc_df[loc_df['Place_Name'] == loc]['Coordinates'].values[0]
            if location:
                location_route_with_coordinates.append(location)
            else:
                location_route_with_coordinates.append(None)
        else:
            location_route_with_coordinates.append(loc)

    st.write('\n')

    rows = []
    distance_total = 0
    initial_loc = ''  # starting point
    location_route_names = []  # list of final route place names in order

    for i, loc in enumerate(location_route_with_coordinates[:-1]):
        next_loc = location_route_with_coordinates[i + 1]

        # Calculate the geodesic distance between two locations
        distance = geodesic(loc, next_loc).kilometers
        distance_km_text = f"{distance:.2f} km"
        distance_mi_text = f"{distance*0.621371:.2f} mi"

        a = loc_df[loc_df['Coordinates'] == loc]['Place_Name'].reset_index(drop=True)[0]
        b = loc_df[loc_df['Coordinates'] == next_loc]['Place_Name'].reset_index(drop=True)[0]
        
        if i == 0:
            location_route_names.append(a.replace(' ', '+') + '/')
            initial_loc = (a.replace(' ', '+')) + '/'
        else:
            location_route_names.append(a.replace(' ', '+') + '/')

        distance_total += distance
        rows.append((a, b, distance_km_text, distance_mi_text))

    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 27px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>⬇️ Here is your Optimal Geodesic Distance 🗺️</i></h5>", unsafe_allow_html=True)
    distance_total = int(round(distance_total*0.621371, 0))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("", '{} miles'.format(distance_total))
        
    df = pd.DataFrame(rows, columns=["From", "To", "Distance (km)", "Distance (mi)"]).reset_index(drop=True)
    
    st.dataframe(df, use_container_width=True)  # display route with distance
    location_route_names.append(initial_loc)
    return location_route_names    

def tsp_solver(data_model, iterations=1000, temperature=10000, cooling_rate=0.95):
    def distance(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    num_locations = data_model['num_locations']
    locations = [(float(lat), float(lng)) for lat, lng in data_model['locations']]

    # Randomly generate a starting solution
    current_solution = list(range(num_locations))
    random.shuffle(current_solution)

    # Compute the distance of the starting solution
    current_distance = 0
    for i in range(num_locations):
        current_distance += distance(locations[current_solution[i-1]], locations[current_solution[i]])

    # Initialize the best solution as the starting solution
    best_solution = current_solution
    best_distance = current_distance

    # Simulated Annealing algorithm
    for i in range(iterations):
        # Compute the temperature for this iteration
        current_temperature = temperature * (cooling_rate ** i)

        # Generate a new solution by swapping two random locations
        new_solution = current_solution.copy()
        j, k = random.sample(range(num_locations), 2)
        new_solution[j], new_solution[k] = new_solution[k], new_solution[j]

        # Compute the distance of the new solution
        new_distance = 0
        for i in range(num_locations):
            new_distance += distance(locations[new_solution[i-1]], locations[new_solution[i]])

        # Decide whether to accept the new solution
        delta = new_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / current_temperature):
            current_solution = new_solution
            current_distance = new_distance

        # Update the best solution if the current solution is better
        if current_distance < best_distance:
            best_solution = current_solution
            best_distance = current_distance

    # Convert the solution to the required format
    x = {}
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                if (i, j) in x:
                    continue
                if (j, i) in x:
                    continue
                if (i == 0 and j == num_locations - 1) or (i == num_locations - 1 and j == 0):
                    x[i, j] = 1
                    x[j, i] = 1
                elif i < j:
                    x[i, j] = 1
                    x[j, i] = 0
                else:
                    x[i, j] = 0
                    x[j, i] = 1

    # Create the optimal route
    optimal_route = []
    start_index = best_solution.index(0)
    for i in range(num_locations):
        optimal_route.append(best_solution[(start_index+i)%num_locations])
    optimal_route.append(0)
    
    # Return the optimal route
    location_route = [locations[i] for i in optimal_route]
    return location_route, x

# Caching the distance matrix calculation for better performance
@st.cache_data
def compute_distance_matrix(locations):    
    # using geopy geodesic for lesser compute time
    num_locations = len(locations)
    distance_matrix = [[0] * num_locations for i in range(num_locations)]
    for i in range(num_locations):
        for j in range(i, num_locations):
            distance = geodesic(locations[i], locations[j]).km
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

def create_data_model(locations):
    data = {}
    num_locations = len(locations)
    data['locations']=locations
    data['num_locations'] = num_locations
    distance_matrix = compute_distance_matrix(locations)
    data['distance_matrix'] = distance_matrix
    return data

def geocode_address(address):
    url = f'https://photon.komoot.io/api/?q={address}'
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if results['features']:
            first_result = results['features'][0]
            latitude = first_result['geometry']['coordinates'][1]
            longitude = first_result['geometry']['coordinates'][0]
            return address, latitude, longitude
        else:
            print(f'Geocode was not successful. No results found for address: {address}')
    else:
        print('Failed to get a response from the geocoding API.')

def main():


# Integrating distance calculation code into the existing Streamlit application

    st.markdown(
        f"<div style='text-align: center; padding: 5px;'><h1 style='font-size: 75px'>Travel Route Planner</h1><h3 style='font-size: 35px;'><i>Journey Optimization with Knapsack & Traveling Salesman Techniques</i></h3></div>",unsafe_allow_html=True)
    
    default_locations = [['Coimbatore'],['Chennai'],['Bangalore'],['Mangalore']]
    existing_locations = '\n'.join([x[0] for x in default_locations])
    
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 27px;color: #3b3b3b;'><hr style='height: 3px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i> Hey There 👋🏻 Where are you heading to? Type below ⬇️</i></h5>", unsafe_allow_html=True)
    selected_value = st.text_area("", value=existing_locations, height=215)
    st.markdown(
    """
    <style>
    textarea {
        font-size: 1.3rem !important;
        font-family:sans-serif !important;
        font-weight: lighter !important;
        background-color: #fdf0ff !important;
        color: #ab43e3 !important;
        line-height: 45px !important;        
    }
    button {
        min-height: 45px !important;
        width: 100% !important;
    }
    p{
        font-size: 1.5rem !important;
        
    }
    .st-emotion-cache-mubdbw:active {
        color: #ffffff !important;
        border-color: #ab43e3 !important;
        background-color: #ab43e3 !important;
    }
    .st-emotion-cache-mubdbw:focus:not(:active) {
        border-color: #ab43e3 !important;
        color: #ab43e3 !important;
    }
    .st-emotion-cache-mubdbw:hover {
        border-color: #c982ef !important;
        color: #3b3b3b;
    }
    .st-emotion-cache-1q9f7mt {
         width: 295px;
        position: relative;
    }
    
    .st-emotion-cache-xvfsap a {
    color: #ab43e3 !important;
    text-decoration: underline !important;
    }
    
    h4 {
    font-style: italic;
    font-size: 27px;
    }
    
    .st-au {
    background-color: #fdf0ff;
    }
    
    .st-al {
    color: #3b3b3b !important;
    }
    
        </style>
    """,
    unsafe_allow_html=True,)
    ('\n')
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 27px;color: #3b3b3b;'><hr style='height: 3px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>⬇️ Fill up to travel light, and prioritize right! 🎒✨</i></h5>", unsafe_allow_html=True)

    # Get inputs from usertest
    ('\n')
    num_items = st.number_input("Enter the number of items:", min_value=1, step=1, value=1)

    weights = []
    values = []
    item_names = []

    for i in range(num_items):
        st.markdown(f"<hr style='height: 1px;background: linear-gradient(to right, #C982EF, #000000)><h5 style='text-align: left; letter-spacing:1px;font-size: 35px !important;color: #3b3b3b;'><b><i style='font-size: 25px'>Item '{i+1}'</i></h5><br><br>", unsafe_allow_html=True)
        item_name = st.text_input(f"Enter item name for item {i+1}:")
        item_names.append(item_name)
        weight = st.number_input(f"Enter weight (in kg) for item {i+1}:", min_value=1, step=1)
        weights.append(weight)
        value = st.number_input(f"Enter value for item {i+1}:", min_value=0, step=1)
        values.append(value)
    
    
    st.markdown(f"<hr style='height: 1px;background: linear-gradient(to right, #C982EF, #000000)><h5 style='text-align: left; letter-spacing:1px;font-size: 35px !important;color: #3b3b3b;'><b><i style='font-size: 25px'></i></h5>", unsafe_allow_html=True)
    capacity = st.number_input("Enter the capacity of the knapsack:", min_value=1, step=1)
    
    
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 27px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i></i></h5>", unsafe_allow_html=True)
        
    solve_expander = st.expander("Click to reveal! 🎒✨", expanded=False)
    with solve_expander:
        max_value, dp = knapsack_max_value(weights, values, capacity)
        included_items = find_included_items(weights, dp)
    
        st.markdown (f"<h5><i style='font-size:20px;'>Maximum value that can be obtained: '{max_value}'</i></h5>",unsafe_allow_html=True)
        st.markdown(
        f"<hr style='height: 1px;background: linear-gradient(to right, #C982EF, #000000);'><b style='font-size: larger'>⬇️ Here are the essentials for your journey! 🎒</b><br><br>", unsafe_allow_html=True)
        for item_idx in included_items:
            #st.write(f"Item {item_idx}: {item_names[item_idx-1]} (Weight: {weights[item_idx-1]}, Value: {values[item_idx-1]})")
            st.markdown(f"<h3 style='font-size: larger; font-weight:lighter'> Item {item_idx}: {item_names[item_idx-1]} (Weight: {weights[item_idx-1]}, Value: {values[item_idx-1]})</h3>",unsafe_allow_html=True)        
            


    if st.button("Discover Efficient Route", type='primary'):
        lines = selected_value.split('\n')
        values = [geocode_address(line) for line in lines if line.strip()]    
        location_names=[x[0] for x in values if x is not None] # address names
        locations=[(x[1],x[2]) for x in values if x is not None] # coordinates        
        loc_df = pd.DataFrame({'Coordinates': locations, 'Place_Name': location_names})    
        
        if locations:
            data_model = create_data_model(locations)
            solution, x = tsp_solver(data_model)

            if solution:
                distance_matrix = compute_distance_matrix(locations)
                location_route_names = display_route(solution, x, locations, loc_df, distance_matrix)
                
                gmap_search = 'https://www.google.com/maps/dir/+'
                gmap_places = gmap_search + ''.join(location_route_names)
                
                st.write('\n')
                st.write("<div style='text-align: center; font-size:1.5rem;'><a href='{}'>→ Unveil Your Optimal Routes here on Google Maps! </a></div>".format(gmap_places), unsafe_allow_html=True)
                # Visualize the undirected weighted graph using NetworkX
                G = nx.Graph()
                for i, loc in enumerate(locations):
                    G.add_node(i, pos=loc)
                
                for i in range(len(locations)):
                    for j in range(i + 1, len(locations)):
                        distance = geodesic(locations[i], locations[j]).kilometers
                        G.add_edge(i, j, weight=distance)
                
                pos = nx.get_node_attributes(G, 'pos')
                edge_labels = {(i, j): f"{d['weight']:.2f} km" for i, j, d in G.edges(data=True)}
                nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                plt.title('Undirected Weighted Graph for TSP')
                st.pyplot(plt)
                st.subheader("Optimal Path:")
                st.write("The optimal path is:", location_route_names)

            
            else:
                st.error("No solution found.")
                
            
    st.write('\n')
    st.write('\n')
    st.write('\n')
    
    
    st.markdown(
        f"<hr style='height: 3px;background: linear-gradient(to right, #C982EF, #b8b8b8);'>", unsafe_allow_html=True)
    st.write('\n')
    st.write('\n')

    
    st.title("🌍 Distance Calculator - Check with AI")

    # Input locations from the user
    locations_input = st.text_area("📍 Enter locations (one per line):")

    if st.button("🚀 Calculate Distance"):
        locations = [location.strip() for location in locations_input.split("\n") if location.strip()]
        if len(locations) < 2:
            st.error("⚠️ Please provide at least two locations.")
        else:
            # Generate the distance calculation using GPT-3.5
            distance_output = generate_distance(locations)
            st.subheader("📏 Distance Calculation:")
            st.code(distance_output, language='text')

    st.write('#### **About**')
    st.write('\n')  
    st.info(
     """
                Created by:
                [Kim](https://in.linkedin.com/in/kimberlymarcelinnathan), [Tanu](https://in.linkedin.com/in/tanushree-rajan-97519420b) (M.sc DCS - Students @ CIT)
            """)
    
if __name__ == "__main__":
    main()
