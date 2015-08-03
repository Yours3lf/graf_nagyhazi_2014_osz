//startup

$(function()
{
	//for in-browser dev
  //init_js();
}
);

function init_js()
{
  $('button').button();
	$(document).tooltip();

  //var duration = 250;
  var duration = 3000;
  $('#intro').fadeIn(duration, function()
  {
    $('#intro').fadeOut(duration, function()
    {
      $(document.body).animate({ "background-color": "rgba(0, 0, 0, 0)" }, duration);
      $('#content').fadeIn(duration);
    });
  });
}
