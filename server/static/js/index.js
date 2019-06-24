$(function() {

  $(document).on('change', ':file', function() {
     // read in name of uploaded file and trigger the fileselect event
      var input = $(this),
          label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
      input.trigger('fileselect', label);

      // read in content of uploaded file and trigger preview plot
      if (this.files && this.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
          console.log(typeof reader.result);
          annotationsObject = JSON.parse(reader.result);
          //process example
          preview_data = {}
          for (var key in annotationsObject){
            preview_data[key] = eval(annotationsObject[key])
          };    
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
  // Loading spinner
  $('#uploadButton').on('click', function() {
    var $this = $(this);
    var loadingText = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    if ($(this).html() !== loadingText) {
      $this.data('original-text', $(this).html());
      $this.html(loadingText);
    }
  });
  // trigger preview plot and scrolling
  $('#previewButton').on('click', function() {
      preview_plot(preview_data);
  });

})


function preview_plot(json_data) {
  var qdlin = { x: Array(1000).fill().map((x,i)=>i), // create an array of range(1000)
                y: json_data["Qdlin"][0][0].flat(), //[0][0] to get first item of batch and first item of window
                mode: 'line',
                name: 'Qdlin', };
  var tdlin = { x: Array(1000).fill().map((x,i)=>i), // create an array of range(1000)
                y: json_data["Tdlin"][0][0].flat(), //[0][0] to get first item of batch and first item of window
                mode: 'line',
                xaxis: 'x2',
                yaxis: 'y2',
                name: 'Tdlin', };
  var data = [ qdlin, tdlin ];
  var layout = {
    plot_bgcolor: 'rgb(250, 250, 250)',
    grid: {rows: 2, columns: 1, pattern: 'independent'},
  };
  $('#previewContainer').append('<div class="container h-100">\
    <div class="row h-100 justify-content-center align-items-center">\
    <div class="col-9 center-block"><div id="preview">\
    </div></div></div></div>');
  Plotly.newPlot('preview', data, layout);
  $('#preview').prepend("<h2>Your Uploaded Data</h2>");
  var ir = parseFloat(json_data['IR'][0][0]).toFixed(2);
  var discharge_time = parseFloat(json_data['Discharge_time'][0][0]).toFixed(2);
  var qd = parseFloat(json_data['QD'][0][0]).toFixed(2);
  var scalars = `<div>IR: ${ir} <br> Discharge Time: ${discharge_time} <br> QD: ${qd}</div><br>`;
  $('#preview').append(scalars);
  $('#preview').append('<button type="button" class="btn btn-primary" id="backToTopButton">Back to top</button>');
  $('#backToTopButton').on('click', (function () {
    $('body,html').animate({
      scrollTop: 0
    }, 800);
    return false;
  }));   
  document.querySelector('#preview').scrollIntoView({
    behavior: 'smooth',
    alignTo: true,
  });
}

