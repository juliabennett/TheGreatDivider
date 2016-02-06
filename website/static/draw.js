
$(document).ready(function(){

  $('#enter').submit(populate);
  $('#dropdown').change(populate);

  function populate() {

    if ($(this).attr('id') == 'enter') {
      var val = $(this).find('input').val();
    }
    else {
      var val = $(this).val();
    }
    
    $('<h2 id="question"></h2>')
      .html('Conversations about &ldquo;' + val + '"?')
      .prependTo(".content");

    $('#lefty').empty();
    $('#righty').empty();
    $('#safe').empty();
    $('.toggle').toggleClass('hidden');

    $.getJSON('/search?keyword='  + val, respond);
    return false;
  }

  function respond(res) {
      $('#loading').addClass('hidden');
      $('#titles').removeClass('hidden');
      $('#again').removeClass('hidden');

      var text = {
        'few': "<div class='sorry'>Sorry, not enough discussion about this topic.</div>", 
        'bad': "<div class='sorry'>Sorry, failed to extract topics. </div>"
      }

      if (res['success'] !== true) {
        $('.clouds').remove();
        $('<h3>' + text[res['success']] + '</h3>')
          .appendTo($('#safe'));
        return false;
      } 

      for (var i=0; i<res['left'].length; i++) {
        draw(res['left'][i], "#lefty", i);
      }

      for (var i=0; i <res['right'].length; i++) { 
        draw(res['right'][i], "#righty", i);
      }

      $(".cloud").click(function(){ return showModal($(this), res);} )
    }  

  function draw(words, side, ind) {
    var cloud = $("<div class='cloud'></div>");

    cloud
      .addClass(ind.toString())
      .appendTo(side);

    for (var i=0; i<words.length; i++) {
      $('<div></div>')
        .html(words[i])
        .appendTo(cloud);  
    }
  }

  function showModal(current, res) {

    $('#links').empty();
    
    var num = current.attr('class').split(/\s+/)[1], 
        side = current.parent().attr("id");

    if (side == "lefty") {
      var urls = res['left_urls'];
    }
    if (side == "righty") {
      var urls = res['right_urls'];
    }
    
    var url_list = urls[Number.parseInt(num)];

    for (var i=0; i<url_list.length; i++) {

      var div = $("<div></div>")
        .appendTo($('#links'));

      var link = 'https://www.reddit.com/' + url_list[i][0], 
          subred = "/r/" + url_list[i][0].split("/")[2].toLowerCase()
          text = url_list[i][1];

      $('<div> <b>' + subred + '</b> </div>')
        .appendTo(div);

      var block = $('<blockquote></blockquote>')
        .appendTo(div);

      $('<div>' + text + '</div>')
        .appendTo(block);

      $("<a></a>")
        .attr("href", link)
        .html('<button class="btn btn-default read-more" type="submit">join the conversation</button>')
        .appendTo(block);

      $('<br>').appendTo($('#links'));
    }

    $('#myModal').modal();
  }

})

