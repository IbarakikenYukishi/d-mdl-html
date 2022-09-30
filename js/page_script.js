function getCSVFile(country) {
  var xhr = new XMLHttpRequest();
  xhr.onload = function() {
    createArray(xhr.responseText, country);
  };
  xhr.open("get", "./data/gaussian_alarm_results.csv", true);
  xhr.send(null);
};

function createArray(csvData, country) {
  var tempArray = csvData.split("\n");
  var csvArray = new Array();
  for (var i = 0; i < tempArray.length; i++) {
    csvArray[i] = tempArray[i].split(",");
    var countryArray = tempArray[i].split(",");
    if (countryArray[0] == country) {
      var element = document.getElementById("previewCSVFile");
      var element_str = '<h3>Statistics</h3>'
      element_str += '<b>'
      if (countryArray[10]==""){
        element_str += 'Latest date of change point alert: None<br>';
      } else {
        element_str += 'Latest date of change point alert: ' + countryArray[10] + '<br>';
      }
      if (countryArray[11]==""){
        element_str += 'Latest date of sign alert in the current window: None<br>';
      } else {
        element_str += 'Latest date of sign alert in the current window: ' + countryArray[11] + '<br>';        
      }
      if (countryArray[12]==""){
        element_str += 'Reference date of the next change point: None<br>';
      } else {
        element_str += 'Reference date of the next change point: ' + countryArray[12] + '<br>';        
      }
      console.log(countryArray[10]);
      console.log(countryArray[11]);
      console.log(countryArray[12]);
      element_str += 'Number of change point alerts in the last month: ' + countryArray[6] + '<br>';
      element_str += 'Number of 1st D-MDL sign alerts in the last month: ' + countryArray[7] + '<br>';
      element_str += 'Number of 2nd D-MDL sign alerts in the last month: ' + countryArray[8] + '<br>';
      element_str += '</b>'
      element.innerHTML = element_str;
    }
  }
  console.log(csvArray);
};

function switchPage(page) {

  var element = document.getElementById("main");
  if (page == 'index') {
    var element_str = '<h2>About</h2> \
                    This website shows the result of D-MDL. \
                    The data source is <a href="https://github.com/CSSEGISandData/COVID-19">John Hopkins University.</a> <br> \
                    The alerts are made on the basis of theory of "differential MDL(minimum description length) statistics(D-MDL)". \
                    See the following reference for the details of the methodology and theory. <br> \
                    <a href="https://arxiv.org/abs/2007.15179">https://arxiv.org/abs/2007.15179</a> <br>\
                    In the paper at arXiv, we used the data from <a href="https://www.ecdc.europa.eu/en">European Centre for Disease Prevention and Control (ECDC)</a>, but we currently use the data from John Hopkins University since the data at ECDC changed the data format from on the daily basis to on the weekly basis.\
                    <h2>Gaussian Modeling</h2> \
                    We assume that the number of new cases follows an independent Gaussian model. \
                    <h3>Statistics</h3> \
                    The following six statistics are shown for each country.<br> \
                    <b>1. Latest date of change point alert</b> <br> \
                    A change point indicates a significant increase or decrease in the number of daily new cases.<br>  \
                    <b>2. Latest date of sign alert</b> <br> \
                    We raise alerts when the 1st or 2nd D-MDL exceeds a threshold. Intuitively, the 1st D-MDL shows the velocity of change and the 2nd D-MDL shows the acceleration of change.<br> \
                    <b>3. Reference date of the next change point</b><br> \
                    We empirically demonstrate that a sign of change is detected six days earlier than change points on average. Thus the reference date is set as the time point of six days later from the date when the first sign is detected.<br> \
                    <b>4. Number of change point alerts in the last month.</b><br> \
                    <b>5. Number of 1st D-MDL sign alerts in the last month.</b><br> \
                    <b>6. Number of 2nd D-MDL sign alerts in the last month.</b> \
                    <h3>Window Size</h3> \
                    The window size when the on-line change point detection algorithm with adaptive windowing is employed. The time point when the window size is zero is a change point. \
                    <h3>0th D-MDL Change Score</h3> \
                    Change point score that indicates the degree of how likely the time point is a change point when the fixed-sized windowing is employed. \
                    <h3>1st D-MDL Change Score</h3> \
                    Change sign score that indicates the velocity of change. \
                    <h3>2nd MDL Change Score</h3> \
                    Change sign score that indicates the acceleration of change. \
                    <h2>Exponential Modeling</h2> \
                    The exponential model establishes an exponential relation between the growth rate of cumulative cases and time. With R0, the basic reproduction number, R0-1 is proportional to the growth rate. Therefore, the change detected by the exponential modeling indicates the change in the growth rate and hence in the R0. \
                    The blue lines corresponds to the increase of R0, and the red lines corresponds to the decrease of R0. \
                    <br clear="all"> ';
    element.innerHTML = element_str;
  } else {
    var country_name = page.replace('_', ' ');
    var element_str = '<h2>Results of Gaussian Modeling(' + country_name + ')</h2> \
                    <p id="previewCSVFile"></p> \
                    <h3>Daily Confirmed Cases</h3>\
                    <br clear="all"> \
                    <img src="./data/gaussian_figs/' + page + '_case.png" width="700">\
                    <h3>Window Size</h3>\
                    <br clear="all">\
                    <img src="./data/gaussian_figs/' + page + '_window_size.png" width="700">\
                    <h3>0th Order D-MDL Score</h3>\
                    <br clear="all">\
                    <img src="./data/gaussian_figs/' + page + '_0_score.png" width="700">\
                    <h3>1st Order D-MDL Score</h3>\
                    <br clear="all">\
                    <img src="./data/gaussian_figs/' + page + '_1_score.png" width="700">\
                    <h3>2nd Order D-MDL Score</h3>\
                    <br clear="all">\
                    <img src="./data/gaussian_figs/' + page + '_2_score.png" width="700">\
                <h2>Results of Exponential Modeling(' + country_name + ')</h2>\
                    <h3>Cumulative Confirmed Cases</h3>\
                    <br clear="all">\
                    <img src="./data/exponential_figs/' + page + '_case.png" width="700">\
                    <h3>Window Size</h3>\
                    <br clear="all">\
                    <img src="./data/exponential_figs/' + page + '_window_size.png" width="700">\
                    <h3>0th Order D-MDL Score</h3>\
                    <br clear="all">\
                    <img src="./data/exponential_figs/' + page + '_0_score.png" width="700">\
                    <h3>1st Order D-MDL Score</h3>\
                    <br clear="all">\
                    <img src="./data/exponential_figs/' + page + '_1_score.png" width="700">\
                    <h3>2nd Order D-MDL Score</h3>\
                    <br clear="all">\
                    <img src="./data/exponential_figs/' + page + '_2_score.png" width="700">';
    element.innerHTML = element_str;
    getCSVFile(page);
  };
};
