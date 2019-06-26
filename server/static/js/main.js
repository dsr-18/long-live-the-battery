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
  // format json data to create two linegraphs
  var qdlin = { x: Array(1000).fill().map((x,i)=>i), // create an array of range(0,1000)
                y: json_data["Qdlin"][0][0].flat(), //[0][0] to get first item of batch and first item of window
                mode: 'line',
                name: 'Qdlin', };
  var tdlin = { x: Array(1000).fill().map((x,i)=>i),
                y: json_data["Tdlin"][0][0].flat(),
                mode: 'line',
                xaxis: 'x2',
                yaxis: 'y2',
                name: 'Tdlin', };
  var data = [ qdlin, tdlin ];
  var layout = {
    grid: {rows: 2, columns: 1, pattern: 'independent'},
  };
  // read in scalars to insert below graphs
  var ir = parseFloat(json_data['IR'][0][0]).toFixed(2);
  var discharge_time = parseFloat(json_data['Discharge_time'][0][0]).toFixed(2);
  var qd = parseFloat(json_data['QD'][0][0]).toFixed(2);
  
  // create preview div
  $('#previewContainer').append('<div class="container h-100">\
  <div class="row h-100 justify-content-center align-items-center">\
  <div class="col-9 center-block"><div id="preview">\
  </div></div></div></div>');
  // insert graph, headline, and scalar values
  Plotly.newPlot('preview', data, layout);
  $('#preview').prepend("<h2>Your Uploaded Data</h2>"); // using prepend, because Plotly.newplot needs to be inserted first
  var scalars = `<div>IR: ${ir} <br> Discharge Time: ${discharge_time} <br> QD: ${qd}</div><br>`;
  $('#preview').append(scalars);
  // add 'back to top' button
  $('#preview').append('<button type="button" class="btn btn-primary" id="backToTopButton">Back to top</button>');
  $('#backToTopButton').on('click', (function () {
    $('body,html').animate({
      scrollTop: 0
    }, 800);
    return false;
  }));   

}

