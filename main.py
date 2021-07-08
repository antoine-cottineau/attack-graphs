import dash
from ui.layout import generate_layout

app = dash.Dash(__name__)
app.layout = generate_layout()

if __name__ == "__main__":
    app.run_server(debug=True)
