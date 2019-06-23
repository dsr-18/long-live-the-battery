$(function() {

    $(document).on('change', ':file', function() {
      var input = $(this),
          label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
      input.trigger('fileselect', label);
    });
  
    $(document).ready( function() {
        $(':file').on('fileselect', function(event, label) {  
            var input = $(this).parents('.input-group').find(':text');  
            input.val(label);
        });
    });
    
});
  
  
$(document).ready(function() {
    $('#uploadButton').on('click', function() {
      var $this = $(this);
      var loadingText = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
      if ($(this).html() !== loadingText) {
        $this.data('original-text', $(this).html());
        $this.html(loadingText);
      }
    });
  })