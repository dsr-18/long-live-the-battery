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
  
  