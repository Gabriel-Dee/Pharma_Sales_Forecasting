import streamlit as st
import pandas as pd
import plotly.express as px

import streamlit_shadcn_ui as ui
from local_components import card_container

# CONFIGS
CURRENT_YEAR = 2024
PREVIOUS_YEAR = 2023

# Assuming you have a dataset URL or a local path
DATA_URL = "Data/data.csv"  # Update this to the actual dataset path or URL
BAR_CHART_COLOR = "#000000"

# Set Streamlit page configuration
st.set_page_config(page_title="Pharmaceuticals Sales Forecasting", page_icon="📈", layout="wide")
st.title(f"Pharmaceuticals Sales Forecasting", anchor=False)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache_data
def get_and_prepare_data(data):
    df = pd.read_csv(data).assign(
        Date=lambda df: pd.to_datetime(df["Date"]),
        Month=lambda df: df["Date"].dt.month,
        Year=lambda df: df["Date"].dt.year,
    )
    return df

df = get_and_prepare_data(data=DATA_URL)

# Convert 'Date' column to string to avoid JSON serialization issues
df['Date'] = df['Date'].astype(str)

# Aggregate data
total_sales = df['Sales'].sum()
total_products = df['Product Code'].nunique()
total_pharmacies = df['Pharmacy Name'].nunique()

# Display aggregate data in cards
columns = st.columns(3)
with columns[0]:
    ui.metric_card(
        title="Total Sales",
        content=f"${total_sales:,.2f}",
        description="Aggregate sales amount",
        key="total_sales_card",
    )
with columns[1]:
    ui.metric_card(
        title="Total Products",
        content=f"{total_products}",
        description="Unique products",
        key="total_products_card",
    )
with columns[2]:
    ui.metric_card(
        title="Total Pharmacies",
        content=f"{total_pharmacies}",
        description="Unique pharmacies",
        key="total_pharmacies_card",
    )

# View of the dataset
st.subheader("Dataset Overview")

# Add button to toggle between the first five rows and the full dataset
if 'view_full' not in st.session_state:
    st.session_state.view_full = False

if st.button("Toggle View"):
    st.session_state.view_full = not st.session_state.view_full

# Display first five rows or the full dataset based on the button state
if st.session_state.view_full:
    ui.table(data=df, maxHeight=300)
else:
    ui.table(data=df.head(), maxHeight=300)

# Section for selecting analysis type
analysis_type = ui.tabs(
    options=["Sales Trend", "Product Sales Distribution", "Pharmacy Sales Comparison", "Sales Over Time"],  # Update with your specific chart names
    default_value="Sales Trend",
    key="analysis_type",
)

# Dropdown for selecting a pharmacy
pharmacies = df['Pharmacy Name'].unique().tolist()
selected_pharmacy = ui.select("Select a pharmacy:", pharmacies)

# Toggle for selecting the year for visualization
previous_year_toggle = ui.switch(
    default_checked=False, label="Previous Year", key="switch_visualization"
)
visualization_year = PREVIOUS_YEAR if previous_year_toggle else CURRENT_YEAR

# Display the year above the chart based on the toggle switch
st.write(f"**Sales for {visualization_year}**")

# Radio buttons for selecting chart type
chart_type = st.radio("Select chart type:", options=["Bar Chart", "Line Chart"])

# Filter data based on selection for visualization
if analysis_type == "Product Sales Distribution":
    filtered_data = (
        df.query("`Pharmacy Name` == @selected_pharmacy & Year == @visualization_year")
        .groupby("Product Code", dropna=False)["Sales"]
        .sum()
        .reset_index()
    )
elif analysis_type == "Sales Over Time":
    filtered_data = df.groupby(["Year", "Month"], dropna=False)["Sales"].sum().reset_index()
    filtered_data["Date"] = pd.to_datetime(filtered_data[["Year", "Month"]].assign(day=1))
else:
    # Group by month number for Sales Trend
    filtered_data = (
        df.query("`Pharmacy Name` == @selected_pharmacy & Year == @visualization_year")
        .groupby("Month", dropna=False)["Sales"]
        .sum()
        .reset_index()
    )
    # Ensure month column is formatted as two digits for consistency
    filtered_data["Month"] = filtered_data["Month"].apply(lambda x: f"{x:02d}")

# Display the data based on the selected chart type
if analysis_type == "Sales Over Time":
    fig = px.line(filtered_data, x='Date', y='Sales', title='Sales Over Time')
    fig.update_traces(line=dict(color='black'))
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title_x=0.5,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        hovermode='x unified',
        annotations=[
            dict(
                x=filtered_data['Date'][0],
                y=filtered_data['Sales'][0],
                xref="x", yref="y",
                text="Start",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            )
        ]
    )
    fig.update_traces(marker_line_width=2, marker_size=10)
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)
else:
    if chart_type == "Bar Chart":
        vega_spec = {
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "x": {
                    "field": filtered_data.columns[0],
                    "type": "nominal",
                    "axis": {
                        "labelAngle": 0,
                        "title": None,  # Hides the x-axis title
                        "grid": False,  # Removes the x-axis gridlines
                    },
                },
                "y": {
                    "field": "Sales",
                    "type": "quantitative",
                    "axis": {
                        "title": None,  # Hides the y-axis title
                        "grid": False,  # Removes the y-axis gridlines
                    },
                },
                "color": {"value": BAR_CHART_COLOR},
            },
            "data": {
                "values": filtered_data.to_dict("records")  # Convert DataFrame to a list of dictionaries
            },
        }
        with card_container(key="chart"):
            st.vega_lite_chart(vega_spec, use_container_width=True)
    else:
        x_field = filtered_data.columns[0]
        fig = px.line(filtered_data, x=x_field, y='Sales', title=f'Sales Trend for {selected_pharmacy} in {visualization_year}')
        fig.update_traces(line=dict(color='black'))
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            title_x=0.5,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            hovermode='x unified'
        )
        fig.update_traces(marker_line_width=2, marker_size=10)
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)
