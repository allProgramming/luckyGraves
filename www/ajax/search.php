<?php 

  // Your MySql information goes here!
  $host = "";
  $user = "";
  $pass = "";

  $databaseName = "";
  $tableName = "graves";

  $con = new PDO("mysql:host=" . $host . ";dbname=" . $databaseName, $user, $pass);

  $queryStr = "SELECT ref, COUNT(ref) FROM $tableName";
  $firstLike = true;
  $words = $_GET['words'];
  foreach ($words as $word) {
    if ($firstLike) {
	    $queryStr .= " WHERE text ";
  	  $firstLike = false;
	  } else {
  	  $queryStr .= " OR text ";
	  }
  	$queryStr .= "LIKE '%" . $word . "%'";
  }
  $queryStr .= " GROUP BY ref ORDER BY COUNT(ref) DESC";

  $cur = $con->prepare($queryStr);
  $cur->execute();
  $rows = $cur->fetchAll();

  echo json_encode($rows);

?>
