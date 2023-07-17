var gd = document.getElementById('line-approx-plot');

Plotly.newPlot(gd, {
  "data": [
    {
      "type": "scatter",
      "x": [1, 2, 3, 4, 5],
      "y": [1, 2, 3, 4, 5],
      color: 'rgb(212, 202, 205)',
    }
  ],

  "layout": {
    "xaxis": {
        "visible": false
    },
    "yaxis": {
        "visible": false,
        "showgrid": false,
        "zeroline": false
    },
    "paper_bgcolor": 'rgba(0,0,0,0)',
    "plot_bgcolor": 'rgba(0,0,0,0)',
    "plot_color": 'rgba(0,0,0,0)'
  }
});
