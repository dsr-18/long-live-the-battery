$(function() {

  var preview_exists = false;
  // file inputs are only interacted with on the index page, not on 
  // the examples page 
  $(document).on('change', ':file', function() {

    // if a preview exists, reset the previewContainer
    if (preview_exists) {
      $('#previewContainer').remove();
      $('#mainContainer').append('<div id="previewContainer"></div>');
      preview_exists = false;
    };

    // read in name of uploaded file and trigger the fileselect event
    var input = $(this),
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', label);

    // read in content of uploaded file and trigger preview plot
    if (this.files && this.files[0]) {
      var reader = new FileReader();
      reader.onload = function(e) {
        sample_data = JSON.parse(reader.result);
        $('#previewButton').prop("disabled", false);  
        $('#uploadButton').prop("disabled", false);  
      };
      reader.readAsText(this.files[0]);
    }
    });
  
  // change label of text to name of uploaded file
  $(':file').on('fileselect', function(event, label) {  
      var input = $(this).parents('.input-group').find(':text');  
      input.val(label);
  });

  // check if dataStorage element exists -> then we're on the examples page
  // and we need to read in the data on page load instead of from the file input
  if($('#dataStorage').length){
    var sample_data = $('#dataStorage').val();
    if (sample_data !== "None") {
      sample_data = sample_data.replace(/'/g, '"');
      sample_data = JSON.parse(sample_data);
      $('#previewButton').prop("disabled", false);  
      $('#uploadButton').prop("disabled", false);  
    };
  };

  // Loading spinner
  $('#uploadButton').on('click', function() {
    var $this = $(this);
    var loadingText = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    $this.html(loadingText);
  });

  // trigger preview plot and scrolling
  $('#previewButton').on('click', function() {
      if (!preview_exists) {
        // process example
        var preview_data = {}
        for (var key in sample_data){
          preview_data[key] = eval(sample_data[key])
        };    
        preview_plot(preview_data);
        preview_exists = true;
      };

      // scroll to graph
      document.querySelector('#preview').scrollIntoView({
        behavior: 'smooth',
        alignTo: true,
      });
  });

})

// visualize data used to make predictions
function preview_plot(json_data) {
  
  // initialize feature variables
  // [0] selects the first item of the batch
  var qdlin = json_data["Qdlin"][0];
  var tdlin = json_data["Tdlin"][0];
  var ir = json_data['IR'][0];
  var dt = json_data['Discharge_time'][0];
  var qd = json_data['QD'][0];
  var window_size = qdlin.length;

  linear_graph_set = linear_graph(qdlin, tdlin, window_size);
  scalar_graph_set = scalar_graph(ir, dt, qd, window_size);
  
  // create preview div
  $('#previewContainer').append('<div class="row h-100 justify-content-center align-items-center">\
  <div class="col-9 center-block px-md-20" id="preview">\
  </div></div>');

  // insert graphs and headline
  Plotly.newPlot('preview', linear_graph_set[0], linear_graph_set[1]);
  $('#preview').prepend("<h2>Your Uploaded Data</h2>"); // using prepend, because Plotly.newplot needs to be inserted first
  $('#preview').append('<div style="height: 300px;" id="preview_2"></div>');
  Plotly.newPlot('preview_2', scalar_graph_set[0], scalar_graph_set[1]);

  // add 'back to top' button
  $('#preview').append('<button type="button" class="btn btn-primary" id="backToTopButton">Back to top</button>');
  $('#backToTopButton').on('click', (function () {
    $('body,html').animate({
      scrollTop: 0
    }, 800);
    return false;
  }));   

}

function linear_graph(qdlin, tdlin, window_size) {

  var linear_data = [];
  
  // Create trace for Qdlin and set the first value to true
  var j;
  for (j = 0; j < window_size; j++) {
    trace = { x: Array(1000).fill().map((x,i)=>i), // create an array of range(0,1000)
              y: qdlin[j].flat(), 
              mode: 'line',
              name: 'Discharge over time',
              visible: false, };
    linear_data.push(trace);
  };
  linear_data[0]['visible'] = true;

  // Create trace for Tdlin and set the first value to true
  var k;
  for (k = 0; k < window_size; k++) {
    trace = { x: Array(1000).fill().map((x,i)=>i),
              y: tdlin[k].flat(),
              mode: 'line',
              name: 'Temperature over time',
              visible: false,
              xaxis: 'x2',
              yaxis: 'y2',
            };
    linear_data.push(trace);
  };
  linear_data[1+window_size]['visible'] = true;

  // Set up steps for slider
  var steps = [];
  var i;
  for (i = 0; i < linear_data.length-window_size; i++) {
    var step = {
      method: 'restyle',  
      args: ['visible', Array(linear_data.length).fill(false)],
      label: i,
    };
    step['args'][1][i] = true; // Toggle i'th trace for Qdlin to "visible"
    step['args'][1][i+window_size] = true; // Toggle trace for Tdlin to "visible"
    steps.push(step);
  };

  // create slider
  var sliders = [{
    active: 0,
    steps: steps,
    currentvalue: {
      visible: true,
      prefix: 'Cycle ',
      xanchor: 'center',
    },
  }];
          
  // create layout
  var tdlin_max = Math.max(...tdlin.flat());
  var tdlin_min = Math.min(...tdlin.flat());
  var qdlin_max = Math.max(...qdlin.flat());
  var linear_layout = {
    title: 'Linear features',
    grid: {rows: 2, columns: 1, pattern: 'independent'},
    sliders: sliders,
    yaxis: {
      range: [0, qdlin_max*1.05],
    },
    yaxis2: {
      range: [tdlin_min, tdlin_max],
    },
    margin: {
      l:40,
      r:10
      },
  };
  return [linear_data, linear_layout]
}
  
function scalar_graph(ir, dt, qd, window_size) {

  // Create trace for IR
  var ir_data = {
    x: Array(window_size).fill().map((x,i)=>i),
    y: ir.flat(),
    mode: 'line',
    name: 'Internal resistance',
  };

  var dt_data = {
    x: Array(window_size).fill().map((x,i)=>i),
    y: dt.flat(),
    mode: 'line',
    name: 'Discharge time',
    xaxis: 'x2',
    yaxis: 'y2',
  };

  var qd_data = {
    x: Array(window_size).fill().map((x,i)=>i),
    y: qd.flat(),
    mode: 'line',
    name: 'Quantity of discharge',
    xaxis: 'x3',
    yaxis: 'y3',
  };

  var scalar_data = [ir_data, dt_data, qd_data];
  var scalar_layout = {
    title:'Scalar features',
    grid: {rows: 1, columns: 3, pattern: 'independent'},
    margin: {
      l:40,
      r:10
      },
  };
  return [scalar_data, scalar_layout]
}
