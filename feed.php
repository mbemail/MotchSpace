<?php
// Relay for bus-map.html: the transit feed sends no CORS header, so the page reads it via this
// same-site file instead (set FEED = "feed.php"). Fixed URL, so it can't be used as an open proxy.
header("Content-Type: application/x-protobuf");
header("Cache-Control: max-age=10");
$data = @file_get_contents("https://metrolinx.tmix.se/gtfs-realtime-saultstemarie/vehiclepositions.pb");
if ($data === false) { http_response_code(502); exit; }
echo $data;
